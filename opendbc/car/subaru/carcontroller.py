import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, make_tester_present_msg
from opendbc.car.lateral import apply_driver_steer_torque_limits, common_fault_avoidance, apply_std_steer_angle_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.subaru import subarucan
from opendbc.car.subaru.values import DBC, GLOBAL_ES_ADDR, CanBus, CarControllerParams, SubaruFlags

# FIXME: These limits aren't exact. The real limit is more than likely over a larger time period and
# involves the total steering angle change rather than rate, but these limits work well for now
MAX_STEER_RATE = 25  # deg/s
MAX_STEER_RATE_FRAMES = 7  # tx control frames needed before torque can be cut


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.apply_torque_last = 0
    self.apply_angle_last = 0
    self.lat_active_prev = False
    self.angle_yielding = False
    self.angle_yield_frames = 0
    self.angle_yield_calm_frames = 0
    self.angle_conflict_frames = 0

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

  def _lkas_angle_blinker(self, CS) -> bool:
    return bool(getattr(CS.out, "leftBlinker", False) or getattr(CS.out, "rightBlinker", False))

  def _lkas_angle_highway_manual(self, CS) -> bool:
    # No turn signal: treat as a driver path correction, not ALC.
    return CS.out.vEgoRaw >= self.p.LKAS_ANGLE_HWY_SPEED and not self._lkas_angle_blinker(CS)

  def _lkas_angle_immediate_conflict(self, CS, requesting: bool) -> bool:
    """EPS latch / low-speed hard fight — drop Request on the first frame."""
    p = self.p
    meas = CS.out.steeringAngleDeg
    if abs(meas) >= p.LKAS_ANGLE_MAX_MEAS:
      return True
    if requesting and abs(self.apply_angle_last - meas) >= p.LKAS_ANGLE_CMD_MEAS_MAX:
      return True
    if self._lkas_angle_highway_manual(CS):
      return False
    if abs(CS.out.steeringTorque) >= p.LKAS_ANGLE_HAND_YIELD:
      return True
    if abs(CS.out.steeringRateDeg) >= p.LKAS_ANGLE_RATE_YIELD:
      return True
    return False

  def _lkas_angle_hwy_manual_conflict(self, CS) -> bool:
    """Firm sustained highway input without blinker. Debounced by caller."""
    if not self._lkas_angle_highway_manual(CS):
      return False
    p = self.p
    return (abs(CS.out.steeringTorque) >= p.LKAS_ANGLE_HAND_YIELD_HWY or
            abs(CS.out.steeringRateDeg) >= p.LKAS_ANGLE_RATE_YIELD_HWY)

  def _lkas_angle_hard_conflict(self, CS, requesting: bool) -> bool:
    if self._lkas_angle_immediate_conflict(CS, requesting):
      return True
    if not self._lkas_angle_hwy_manual_conflict(CS):
      self.angle_conflict_frames = 0
      return False
    self.angle_conflict_frames += 1
    return self.angle_conflict_frames >= self.p.LKAS_ANGLE_HWY_YIELD_DEBOUNCE

  def _lkas_angle_resume_ok(self, CS) -> bool:
    p = self.p
    hwy_manual = self._lkas_angle_highway_manual(CS)
    hand_ok = p.LKAS_ANGLE_HAND_RESUME_HWY if hwy_manual else p.LKAS_ANGLE_HAND_RESUME
    rate_ok = p.LKAS_ANGLE_RATE_RESUME_HWY if hwy_manual else p.LKAS_ANGLE_RATE_RESUME
    return (abs(CS.out.steeringTorque) <= hand_ok and
            abs(CS.out.steeringRateDeg) <= rate_ok and
            abs(CS.out.steeringAngleDeg) < p.LKAS_ANGLE_MAX_MEAS)

  def handle_angle_lateral(self, CC, CS):
    # Request=latActive unless a hard conflict (hand / rate / large angle /
    # cmd lag). First Request 0→1 TX is always cmd=meas (route 29 panda latch).
    # Desired-measured error is never a resume gate (that deadlocked Request).
    meas = CS.out.steeringAngleDeg
    lat_req = bool(CC.latActive)
    rising_req = False

    if not CC.latActive:
      self.angle_yielding = False
      self.angle_yield_frames = 0
      self.angle_yield_calm_frames = 0
      self.angle_conflict_frames = 0
      lat_req = False
    else:
      if self._lkas_angle_hard_conflict(CS, requesting=self.lat_active_prev):
        if not self.angle_yielding:
          self.angle_yielding = True
          self.angle_yield_frames = 0
          self.angle_yield_calm_frames = 0
        self.angle_yield_frames += 1
        self.angle_yield_calm_frames = 0
        lat_req = False
      elif self.angle_yielding:
        self.angle_yield_frames += 1
        min_hold = self.angle_yield_frames >= self.p.LKAS_ANGLE_YIELD_MIN_FRAMES
        if min_hold and self._lkas_angle_resume_ok(CS):
          self.angle_yield_calm_frames += 1
        else:
          self.angle_yield_calm_frames = 0
        calm_need = (self.p.LKAS_ANGLE_RESUME_CALM_FRAMES_HWY
                     if self._lkas_angle_highway_manual(CS)
                     else self.p.LKAS_ANGLE_RESUME_CALM_FRAMES)
        if min_hold and self.angle_yield_calm_frames >= calm_need:
          self.angle_yielding = False
          self.angle_yield_frames = 0
          self.angle_yield_calm_frames = 0
          rising_req = True
        else:
          lat_req = False
      elif not self.lat_active_prev:
        rising_req = True

    if not lat_req:
      apply_steer = meas
    elif rising_req:
      apply_steer = meas
    else:
      apply_steer = apply_std_steer_angle_limits(
            CC.actuators.steeringAngleDeg,
            self.apply_angle_last,
            CS.out.vEgoRaw,
            meas,
            True,
            self.p.ANGLE_LIMITS
          )

    self.apply_angle_last = apply_steer
    self.lat_active_prev = lat_req
    return subarucan.create_steering_control_angle(self.packer, apply_steer, lat_req)

  def handle_torque_lateral(self, CC, CS):
    apply_torque = int(round(CC.actuators.torque * self.p.STEER_MAX))

    new_torque = int(round(apply_torque))
    apply_torque = apply_driver_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorque, self.p)

    if not CC.latActive:
      apply_torque = 0

    if self.CP.flags & SubaruFlags.PREGLOBAL:
      msg = subarucan.create_preglobal_steering_control(self.packer, self.frame // self.p.STEER_STEP, apply_torque, CC.latActive)
    else:
      apply_steer_req = CC.latActive

      if self.CP.flags & SubaruFlags.STEER_RATE_LIMITED:
        # Steering rate fault prevention
        self.steer_rate_counter, apply_steer_req = \
          common_fault_avoidance(abs(CS.out.steeringRateDeg) > MAX_STEER_RATE, apply_steer_req,
                                 self.steer_rate_counter, MAX_STEER_RATE_FRAMES)

      msg = subarucan.create_steering_control(self.packer, apply_torque, apply_steer_req)

    self.apply_torque_last = apply_torque
    self.lat_active_prev = CC.latActive
    return msg

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel

    can_sends = []

    # *** steering ***
    if (self.frame % self.p.STEER_STEP) == 0:
      if self.CP.flags & SubaruFlags.LKAS_ANGLE:
        can_sends.append(self.handle_angle_lateral(CC, CS))
      else:
        can_sends.append(self.handle_torque_lateral(CC, CS))

    # *** longitudinal ***

    if CC.longActive:
      apply_throttle = int(round(np.interp(actuators.accel, CarControllerParams.THROTTLE_LOOKUP_BP, CarControllerParams.THROTTLE_LOOKUP_V)))
      apply_rpm = int(round(np.interp(actuators.accel, CarControllerParams.RPM_LOOKUP_BP, CarControllerParams.RPM_LOOKUP_V)))
      apply_brake = int(round(np.interp(actuators.accel, CarControllerParams.BRAKE_LOOKUP_BP, CarControllerParams.BRAKE_LOOKUP_V)))

      # limit min and max values
      cruise_throttle = np.clip(apply_throttle, CarControllerParams.THROTTLE_MIN, CarControllerParams.THROTTLE_MAX)
      cruise_rpm = np.clip(apply_rpm, CarControllerParams.RPM_MIN, CarControllerParams.RPM_MAX)
      cruise_brake = np.clip(apply_brake, CarControllerParams.BRAKE_MIN, CarControllerParams.BRAKE_MAX)
    else:
      cruise_throttle = CarControllerParams.THROTTLE_INACTIVE
      cruise_rpm = CarControllerParams.RPM_MIN
      cruise_brake = CarControllerParams.BRAKE_MIN

    # *** alerts and pcm cancel ***
    if self.CP.flags & SubaruFlags.PREGLOBAL:
      if self.frame % 5 == 0:
        # 1 = main, 2 = set shallow, 3 = set deep, 4 = resume shallow, 5 = resume deep
        # disengage ACC when OP is disengaged
        if pcm_cancel_cmd:
          cruise_button = 1
        # turn main on if off and past start-up state
        elif not CS.out.cruiseState.available and CS.ready:
          cruise_button = 1
        else:
          cruise_button = CS.cruise_button

        # unstick previous mocked button press
        if cruise_button == 1 and self.cruise_button_prev == 1:
          cruise_button = 0
        self.cruise_button_prev = cruise_button

        can_sends.append(subarucan.create_preglobal_es_distance(self.packer, cruise_button, CS.es_distance_msg))

    else:
      if self.frame % 10 == 0:
        can_sends.append(subarucan.create_es_dashstatus(self.packer, self.frame // 10, CS.es_dashstatus_msg, CC.enabled,
                                                        self.CP.openpilotLongitudinalControl, CC.longActive, hud_control.leadVisible))

        can_sends.append(subarucan.create_es_lkas_state(self.packer, self.frame // 10, CS.es_lkas_state_msg, CC.enabled, hud_control.visualAlert,
                                                        hud_control.leftLaneVisible, hud_control.rightLaneVisible,
                                                        hud_control.leftLaneDepart, hud_control.rightLaneDepart))

        if self.CP.flags & SubaruFlags.SEND_INFOTAINMENT:
          can_sends.append(subarucan.create_es_infotainment(self.packer, self.frame // 10, CS.es_infotainment_msg, hud_control.visualAlert))

      if self.CP.openpilotLongitudinalControl:
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_status(self.packer, self.frame // 5, CS.es_status_msg,
                                                      self.CP.openpilotLongitudinalControl, CC.longActive, cruise_rpm))

          can_sends.append(subarucan.create_es_brake(self.packer, self.frame // 5, CS.es_brake_msg,
                                                     self.CP.openpilotLongitudinalControl, CC.longActive, cruise_brake))

          can_sends.append(subarucan.create_es_distance(self.packer, self.frame // 5, CS.es_distance_msg, 0, pcm_cancel_cmd,
                                                        self.CP.openpilotLongitudinalControl, cruise_brake > 0, cruise_throttle))
      else:
        if pcm_cancel_cmd:
          if not (self.CP.flags & SubaruFlags.HYBRID):
            bus = CanBus.alt if self.CP.flags & SubaruFlags.GLOBAL_GEN2 else CanBus.main
            can_sends.append(subarucan.create_es_distance(self.packer, CS.es_distance_msg["COUNTER"] + 1, CS.es_distance_msg, bus, pcm_cancel_cmd))

      if self.CP.flags & SubaruFlags.DISABLE_EYESIGHT:
        # Tester present (keeps eyesight disabled)
        if self.frame % 100 == 0:
          can_sends.append(make_tester_present_msg(GLOBAL_ES_ADDR, CanBus.camera, suppress_response=True))

        # Create all of the other eyesight messages to keep the rest of the car happy when eyesight is disabled
        if self.frame % 5 == 0:
          can_sends.append(subarucan.create_es_highbeamassist(self.packer))

        if self.frame % 10 == 0:
          can_sends.append(subarucan.create_es_static_1(self.packer))

        if self.frame % 2 == 0:
          can_sends.append(subarucan.create_es_static_2(self.packer))

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    new_actuators.torque = self.apply_torque_last / self.p.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last

    self.frame += 1
    return new_actuators, can_sends
