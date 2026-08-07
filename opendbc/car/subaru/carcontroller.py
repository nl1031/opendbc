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
    # LKAS_ANGLE engage safety gates (see CarControllerParams)
    self.angle_engage_holdoff = False
    self.angle_hand_yielding = False
    self.angle_yield_frames = 0         # STEER_STEP ticks spent yielding
    self.angle_yield_calm_frames = 0     # consecutive calm ticks while yielding
    self.angle_holdoff_calm_frames = 0   # consecutive calm ticks while holdoff

    self.cruise_button_prev = 0
    self.steer_rate_counter = 0

    self.p = CarControllerParams(CP)
    self.packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])

  def _lkas_angle_conditions_calm(self, CS) -> bool:
    """True when wheel rate/hand are calm enough to assert LKAS_Request."""
    rate = abs(CS.out.steeringRateDeg)
    hand = abs(CS.out.steeringTorque)
    return (rate <= self.p.LKAS_ANGLE_ENGAGE_MAX_RATE and
            hand <= self.p.LKAS_ANGLE_HAND_RESUME)

  def _lkas_angle_resume_ok(self, CS) -> bool:
    """Resume after hand-yield: calm rate + low hand (no |des−meas| gate — route 36)."""
    return self._lkas_angle_conditions_calm(CS)

  def _lkas_angle_holdoff_needed(self, CS) -> bool:
    """Hold off Request=1 when wheel angle large or not calm / hands fighting."""
    meas_abs = abs(CS.out.steeringAngleDeg)
    hand = abs(CS.out.steeringTorque)
    return (meas_abs > self.p.LKAS_ANGLE_ENGAGE_MAX_ANGLE or
            not self._lkas_angle_conditions_calm(CS) or
            hand >= self.p.LKAS_ANGLE_HAND_YIELD)

  def _lkas_angle_holdoff_clear_frames(self, CS) -> int:
    """More calm frames when |meas| is still large (gentler re-engage)."""
    if abs(CS.out.steeringAngleDeg) >= self.p.LKAS_ANGLE_LARGE_ANGLE_DEG:
      return self.p.LKAS_ANGLE_LARGE_ANGLE_CALM_FRAMES
    return self.p.LKAS_ANGLE_RESUME_CALM_FRAMES

  def handle_angle_lateral(self, CC, CS):
    # Outback 2023 LKAS_ANGLE (routes 2e / 30 / 33 / 36):
    # - Hard hand-yield only; min hold + calm debounce (no soft-yield chatter).
    # - Resume without |des−meas| gate (that kept Request=0 too long then snapped).
    # - Request 0→1 first TX: cmd = meas (dA=0), then rate-limit toward des (panda-safe).
    # - Inactive: cmd = meas. Active (after first frame): rate-limit toward des.
    meas = CS.out.steeringAngleDeg
    hand = abs(CS.out.steeringTorque)
    des = CC.actuators.steeringAngleDeg
    lat_req = bool(CC.latActive)
    rising_req = False  # Request 0→1 this tick — force cmd=meas

    if not CC.latActive:
      self.angle_engage_holdoff = False
      self.angle_hand_yielding = False
      self.angle_yield_frames = 0
      self.angle_yield_calm_frames = 0
      self.angle_holdoff_calm_frames = 0
      lat_req = False
    else:
      # 1) Hard hand yield — never fight the driver
      if hand >= self.p.LKAS_ANGLE_HAND_YIELD:
        if not self.angle_hand_yielding:
          self.angle_hand_yielding = True
          self.angle_yield_frames = 1
          self.angle_yield_calm_frames = 0
        else:
          self.angle_yield_frames += 1
          self.angle_yield_calm_frames = 0
      elif self.angle_hand_yielding:
        self.angle_yield_frames += 1
        min_hold = self.angle_yield_frames >= self.p.LKAS_ANGLE_YIELD_MIN_FRAMES
        if min_hold and self._lkas_angle_resume_ok(CS):
          self.angle_yield_calm_frames += 1
        else:
          self.angle_yield_calm_frames = 0
        if (min_hold and
            self.angle_yield_calm_frames >= self.p.LKAS_ANGLE_RESUME_CALM_FRAMES):
          self.angle_hand_yielding = False
          self.angle_yield_frames = 0
          self.angle_yield_calm_frames = 0
          self.apply_angle_last = meas
          rising_req = True  # about to assert Request after yield

      if self.angle_hand_yielding:
        lat_req = False

      # 2) Engage hold-off — large angle / not calm at rising edge
      if self.angle_engage_holdoff:
        can_clear = (CC.latActive and
                     not self.angle_hand_yielding and
                     self._lkas_angle_conditions_calm(CS) and
                     hand < self.p.LKAS_ANGLE_HAND_YIELD and
                     abs(meas) <= self.p.LKAS_ANGLE_ENGAGE_MAX_ANGLE)
        need = self._lkas_angle_holdoff_clear_frames(CS)
        if can_clear:
          self.angle_holdoff_calm_frames += 1
          if self.angle_holdoff_calm_frames >= need:
            self.apply_angle_last = meas
            self.angle_engage_holdoff = False
            self.angle_holdoff_calm_frames = 0
            rising_req = True
          else:
            lat_req = False
        else:
          self.angle_holdoff_calm_frames = 0
          lat_req = False
      elif lat_req and not self.lat_active_prev:
        # Rising edge of latActive path
        self.apply_angle_last = meas
        if self._lkas_angle_holdoff_needed(CS):
          self.angle_engage_holdoff = True
          self.angle_holdoff_calm_frames = 0
          lat_req = False
        else:
          rising_req = True

    if not lat_req:
      # Inactive: TX must match measured (panda inactive angle check).
      apply_steer = meas
    elif rising_req:
      # First Request=1 frame: cmd = meas so dA vs last inactive TX is ~0 (route 36).
      apply_steer = meas
      self.apply_angle_last = meas
    else:
      apply_steer = apply_std_steer_angle_limits(
            des,
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
