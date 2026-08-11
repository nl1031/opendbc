import unittest
from types import SimpleNamespace

from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR


def _cs(angle=0.0, rate=0.0, torque=0.0, v=10.0):
  return SimpleNamespace(out=SimpleNamespace(
    vEgoRaw=v,
    steeringAngleDeg=angle,
    steeringRateDeg=rate,
    steeringTorque=torque,
  ))


def _cc(lat_active=True, des_angle=0.0):
  return SimpleNamespace(
    latActive=lat_active,
    actuators=SimpleNamespace(steeringAngleDeg=des_angle),
  )


class TestSubaruCarController(unittest.TestCase):
  def setUp(self):
    CP = CarInterface.get_non_essential_params(CAR.SUBARU_OUTBACK_2023)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.SUBARU_OUTBACK_2023)
    self.controller = CarController({}, CP, CP_SP)

  def test_lkas_angle_rising_edge_first_frame_is_meas(self):
    """Request 0→1 first TX must be meas (dA=0), not a rate step toward des."""
    self.controller.apply_angle_last = 2.46

    cs = _cs(angle=2.61, rate=0.0, torque=0.0, v=8.791)
    cc = _cc(lat_active=True, des_angle=1.11)

    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 2.61)

    # Second active frame rate-limits toward des from meas
    cs = _cs(angle=2.61, rate=0.0, torque=0.0, v=8.791)
    cc = _cc(lat_active=True, des_angle=1.11)
    self.controller.handle_angle_lateral(cc, cs)
    expected = apply_std_steer_angle_limits(
      1.11, 2.61, 8.791, 2.61, True, self.controller.p.ANGLE_LIMITS,
    )
    self.assertAlmostEqual(self.controller.apply_angle_last, expected)

  def test_lkas_angle_holdoff_large_angle_on_engage(self):
    """Do not assert LKAS_Request when engaging at large wheel angle."""
    cs = _cs(angle=44.0, rate=5.0, torque=10.0, v=8.0)
    cc = _cc(lat_active=True, des_angle=20.0)

    msg = self.controller.handle_angle_lateral(cc, cs)
    # ES_LKAS_ANGLE packed values: LKAS_Request should be 0
    self.assertTrue(self.controller.angle_engage_holdoff)
    self.assertFalse(self.controller.lat_active_prev)
    # apply tracks measured while held off
    self.assertAlmostEqual(self.controller.apply_angle_last, 44.0)

  def test_lkas_angle_far_target_engages_from_measured(self):
    """A far target must not deadlock Request; first command still starts at measured."""
    cs = _cs(angle=0.0, rate=0.0, torque=0.0, v=25.0)
    cc = _cc(lat_active=True, des_angle=5.0)

    self.controller.handle_angle_lateral(cc, cs)
    self.assertFalse(self.controller.angle_engage_holdoff)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.0)

    # The next TX approaches the far target only by the highway rate cap.
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.15, places=3)

  def test_lkas_angle_hand_yield(self):
    """High hand torque yields LKAS_Request even after a calm engage."""
    # Calm engage first
    cs = _cs(angle=5.0, rate=1.0, torque=10.0, v=10.0)
    cc = _cc(lat_active=True, des_angle=5.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)

    # Hands fight (above HAND_YIELD=55)
    cs = _cs(angle=8.0, rate=10.0, torque=100.0, v=10.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.angle_hand_yielding)
    self.assertFalse(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 8.0)

  def test_lkas_angle_no_soft_yield(self):
    """Soft yield removed — residual torque alone must not drop Request."""
    cs = _cs(angle=5.0, rate=1.0, torque=10.0, v=10.0)
    cc = _cc(lat_active=True, des_angle=5.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)

    # st=35 < HAND_YIELD=55, |des-meas|=1.7 — previously soft-yielded; must stay active
    cs = _cs(angle=4.3, rate=0.0, torque=35.0, v=7.0)
    cc = _cc(lat_active=True, des_angle=2.6)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertFalse(self.controller.angle_hand_yielding)
    self.assertTrue(self.controller.lat_active_prev)

  def test_lkas_angle_yield_min_hold_blocks_fast_resume(self):
    """Cannot resume Request within YIELD_MIN_FRAMES (anti-chatter)."""
    cs = _cs(angle=5.0, rate=0.0, torque=10.0, v=10.0)
    cc = _cc(lat_active=True, des_angle=5.0)
    self.controller.handle_angle_lateral(cc, cs)

    # Enter yield
    cs = _cs(angle=8.0, rate=5.0, torque=100.0, v=10.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.angle_hand_yielding)

    # Hands off immediately + calm — still held by min frames
    min_frames = self.controller.p.LKAS_ANGLE_YIELD_MIN_FRAMES
    for i in range(min_frames - 1):
      cs = _cs(angle=5.0, rate=0.0, torque=10.0, v=10.0)
      cc = _cc(lat_active=True, des_angle=5.0)
      self.controller.handle_angle_lateral(cc, cs)
      self.assertTrue(self.controller.angle_hand_yielding,
                      f"resumed too early at yield_frame={i+1}")

  def test_lkas_angle_yield_resumes_after_hold_and_calm(self):
    """Resume after min hold + consecutive calm frames even when target is far."""
    cs = _cs(angle=5.0, rate=0.0, torque=10.0, v=10.0)
    cc = _cc(lat_active=True, des_angle=5.0)
    self.controller.handle_angle_lateral(cc, cs)

    cs = _cs(angle=8.0, rate=5.0, torque=100.0, v=10.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.angle_hand_yielding)

    min_frames = self.controller.p.LKAS_ANGLE_YIELD_MIN_FRAMES
    calm_need = self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES
    # Desired/measured error does not gate Request; first-frame locking and slew do.
    for _ in range(min_frames + calm_need + 2):
      if not self.controller.angle_hand_yielding:
        break
      cs = _cs(angle=12.0, rate=0.0, torque=10.0, v=10.0)
      cc = _cc(lat_active=True, des_angle=0.0)
      self.controller.handle_angle_lateral(cc, cs)

    self.assertFalse(self.controller.angle_hand_yielding)
    self.assertTrue(self.controller.lat_active_prev)
    # First resumed TX is meas (dA=0)
    self.assertAlmostEqual(self.controller.apply_angle_last, 12.0)

  def test_lkas_angle_holdoff_clears_when_calm(self):
    """After large-angle hold-off, resume after consecutive calm frames."""
    cs = _cs(angle=40.0, rate=5.0, torque=10.0, v=8.0)
    cc = _cc(lat_active=True, des_angle=10.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.angle_engage_holdoff)

    cs = _cs(angle=30.0, rate=5.0, torque=10.0, v=8.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.angle_engage_holdoff)
    self.assertFalse(self.controller.lat_active_prev)

    # |meas|=10 < LARGE_ANGLE → RESUME_CALM_FRAMES; target distance is irrelevant.
    calm_need = self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES
    for _ in range(calm_need + 2):
      cs = _cs(angle=10.0, rate=2.0, torque=10.0, v=8.0)
      cc = _cc(lat_active=True, des_angle=-10.0)
      self.controller.handle_angle_lateral(cc, cs)
      if not self.controller.angle_engage_holdoff:
        # First Request=1 frame after holdoff: cmd = meas
        self.assertTrue(self.controller.lat_active_prev)
        self.assertAlmostEqual(self.controller.apply_angle_last, 10.0)
        break
    else:
      self.fail("holdoff never cleared")

  def test_lkas_angle_active_rate_limited(self):
    """After first-frame meas lock, active path respects speed-dependent °/TX."""
    # High speed → 0.15°/TX cap
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=35.0)
    cc = _cc(lat_active=True, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.0)

    cs = _cs(angle=0.2, rate=0.0, torque=5.0, v=35.0)
    cc = _cc(lat_active=True, des_angle=20.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.15, places=3)

    # Low speed → 3.5°/TX
    self.controller.lat_active_prev = False
    self.controller.apply_angle_last = 0.0
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=0.0)
    cc = _cc(lat_active=True, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=0.0)
    cc = _cc(lat_active=True, des_angle=20.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 3.5, places=2)

    # Mid speed (5 m/s) → 1.0°/TX
    self.controller.lat_active_prev = False
    self.controller.apply_angle_last = 0.0
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=5.0)
    cc = _cc(lat_active=True, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=5.0)
    cc = _cc(lat_active=True, des_angle=20.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 1.0, places=2)

  def test_lkas_angle_resume_with_far_target_starts_at_measured(self):
    """A far planner target resumes safely instead of leaving Request off forever."""
    self.controller.handle_angle_lateral(_cc(True, 5.0), _cs(5.0, 0.0, 10.0, 10.0))
    self.controller.handle_angle_lateral(_cc(True, 5.0), _cs(8.0, 0.0, 100.0, 10.0))
    self.assertTrue(self.controller.angle_hand_yielding)

    n = (self.controller.p.LKAS_ANGLE_YIELD_MIN_FRAMES +
         self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES + 10)
    for _ in range(n):
      if not self.controller.angle_hand_yielding:
        break
      self.controller.handle_angle_lateral(
        _cc(True, 0.0), _cs(angle=8.0, rate=0.0, torque=10.0, v=10.0))

    self.assertFalse(self.controller.angle_hand_yielding)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 8.0)

  def test_lkas_angle_repeat_override_uses_finite_calm_cooldown(self):
    """A second correction pauses control, then retries without cycling latActive."""
    self.controller.handle_angle_lateral(_cc(True, 1.0), _cs(1.0, 0.0, 10.0, 25.0))
    self.controller.handle_angle_lateral(_cc(True, 1.0), _cs(2.0, 0.0, 100.0, 25.0))

    n = (self.controller.p.LKAS_ANGLE_YIELD_MIN_FRAMES +
         self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES + 2)
    for _ in range(n):
      if not self.controller.angle_hand_yielding:
        break
      self.controller.handle_angle_lateral(
        _cc(True, 2.0), _cs(angle=2.0, rate=0.0, torque=10.0, v=25.0))

    self.assertFalse(self.controller.angle_hand_yielding)
    self.assertGreater(self.controller.angle_resume_monitor_frames, 0)

    self.controller.handle_angle_lateral(
      _cc(True, 2.0), _cs(angle=2.0, rate=0.0, torque=100.0, v=25.0))
    cooldown = self.controller.p.LKAS_ANGLE_RECONFLICT_COOLDOWN_FRAMES
    self.assertEqual(self.controller.angle_reconflict_cooldown_frames, cooldown)
    self.assertFalse(self.controller.lat_active_prev)

    # Non-calm input restarts the finite cooldown.
    for _ in range(5):
      self.controller.handle_angle_lateral(
        _cc(True, 20.0), _cs(angle=2.0, rate=20.0, torque=0.0, v=25.0))
    self.assertEqual(self.controller.angle_reconflict_cooldown_frames, cooldown)
    self.assertFalse(self.controller.lat_active_prev)

    for _ in range(cooldown - 1):
      self.controller.handle_angle_lateral(
        _cc(True, 20.0), _cs(angle=2.0, rate=0.0, torque=0.0, v=25.0))
    self.assertEqual(self.controller.angle_reconflict_cooldown_frames, 1)
    self.assertFalse(self.controller.lat_active_prev)

    self.controller.handle_angle_lateral(
      _cc(True, 20.0), _cs(angle=2.0, rate=0.0, torque=0.0, v=25.0))
    self.assertEqual(self.controller.angle_reconflict_cooldown_frames, 0)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 2.0)


if __name__ == "__main__":
  unittest.main()
