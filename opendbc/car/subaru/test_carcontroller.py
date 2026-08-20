import unittest
from types import SimpleNamespace

from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.subaru.carcontroller import CarController
from opendbc.car.subaru.interface import CarInterface
from opendbc.car.subaru.values import CAR


def _cs(angle=0.0, rate=0.0, torque=0.0, v=10.0, left_blinker=False, right_blinker=False):
  return SimpleNamespace(out=SimpleNamespace(
    vEgoRaw=v,
    steeringAngleDeg=angle,
    steeringRateDeg=rate,
    steeringTorque=torque,
    leftBlinker=left_blinker,
    rightBlinker=right_blinker,
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
    """Request 0→1 first TX is meas (dA=0); next TX uses the JacobW rate table."""
    self.controller.apply_angle_last = 2.46

    cs = _cs(angle=2.61, rate=0.0, torque=0.0, v=8.791)
    cc = _cc(lat_active=True, des_angle=1.11)

    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 2.61)

    self.controller.handle_angle_lateral(cc, cs)
    expected = apply_std_steer_angle_limits(
      1.11, 2.61, 8.791, 2.61, True, self.controller.p.ANGLE_LIMITS,
    )
    self.assertAlmostEqual(self.controller.apply_angle_last, expected)

  def test_lkas_angle_rising_edge_no_overstep_at_10ms(self):
    """Reproduce route 29: first step must not exceed ~0.70°/TX at 9.8 m/s."""
    self.controller.lat_active_prev = False
    self.controller.apply_angle_last = -8.99
    cs = _cs(angle=-8.86, rate=6.5, torque=3.0, v=9.79)
    cc = _cc(lat_active=True, des_angle=-7.62)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, -8.86)
    self.assertLess(abs(self.controller.apply_angle_last - (-8.99)), 0.70)

  def test_light_hand_keeps_request(self):
    """Torque 100 used to feel like OP was weak if it yielded at 55."""
    cs = _cs(angle=5.0, rate=1.0, torque=10.0, v=10.0)
    cc = _cc(lat_active=True, des_angle=5.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)

    cs = _cs(angle=5.2, rate=4.0, torque=100.0, v=10.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertFalse(self.controller.angle_yielding)

  def test_hard_hand_yields_request(self):
    """Route 2e: torque ~190–260 must drop Request and cmd=meas."""
    self.controller.handle_angle_lateral(_cc(True, 5.0), _cs(5.0, 1.0, 10.0, 10.0))
    self.controller.handle_angle_lateral(_cc(True, 8.0), _cs(-40.0, 80.0, 200.0, 4.5))
    self.assertTrue(self.controller.angle_yielding)
    self.assertFalse(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, -40.0)

  def test_high_rate_yields_request(self):
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 10.0, 10.0))
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(-10.0, 80.0, 20.0, 5.0))
    self.assertTrue(self.controller.angle_yielding)
    self.assertFalse(self.controller.lat_active_prev)

  def test_large_angle_yields_request(self):
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 10.0, 10.0))
    self.controller.handle_angle_lateral(_cc(True, 20.0), _cs(50.0, 5.0, 20.0, 5.0))
    self.assertTrue(self.controller.angle_yielding)
    self.assertFalse(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 50.0)

  def test_cmd_meas_gap_yields_request(self):
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 10.0, 10.0))
    self.controller.apply_angle_last = -128.0
    self.controller.lat_active_prev = True
    self.controller.handle_angle_lateral(_cc(True, -128.0), _cs(-179.0, 20.0, 40.0, 5.0))
    self.assertTrue(self.controller.angle_yielding)
    self.assertFalse(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, -179.0)

  def test_far_desired_does_not_block_resume(self):
    """No |des-meas| resume gate — that deadlocked Request while cmd=meas."""
    self.controller.handle_angle_lateral(_cc(True, 5.0), _cs(5.0, 0.0, 10.0, 10.0))
    self.controller.handle_angle_lateral(_cc(True, 5.0), _cs(8.0, 0.0, 200.0, 10.0))
    self.assertTrue(self.controller.angle_yielding)

    n = (self.controller.p.LKAS_ANGLE_YIELD_MIN_FRAMES +
         self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES + 5)
    for _ in range(n):
      if not self.controller.angle_yielding:
        break
      self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(8.0, 0.0, 10.0, 10.0))

    self.assertFalse(self.controller.angle_yielding)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 8.0)

  def test_lkas_angle_inactive_tracks_measured(self):
    cs = _cs(angle=12.3, rate=0.0, torque=0.0, v=20.0)
    cc = _cc(lat_active=False, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertFalse(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 12.3)

  def test_lkas_angle_far_target_rate_limited(self):
    cs = _cs(angle=0.0, rate=0.0, torque=0.0, v=25.0)
    cc = _cc(lat_active=True, des_angle=5.0)

    self.controller.handle_angle_lateral(cc, cs)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.0)

    self.controller.handle_angle_lateral(cc, cs)
    expected = apply_std_steer_angle_limits(
      5.0, 0.0, 25.0, 0.0, True, self.controller.p.ANGLE_LIMITS,
    )
    self.assertAlmostEqual(self.controller.apply_angle_last, expected)
    self.assertGreater(expected, 0.30)
    self.assertLess(expected, 0.45)

  def test_lkas_angle_active_rate_limited(self):
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=35.0)
    cc = _cc(lat_active=True, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.0)

    cs = _cs(angle=0.2, rate=0.0, torque=5.0, v=35.0)
    cc = _cc(lat_active=True, des_angle=20.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.15, places=3)

    self.controller.lat_active_prev = False
    self.controller.apply_angle_last = 0.0
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=0.0)
    cc = _cc(lat_active=True, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=0.0)
    cc = _cc(lat_active=True, des_angle=20.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 5.0, places=2)

    self.controller.lat_active_prev = False
    self.controller.apply_angle_last = 0.0
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=5.0)
    cc = _cc(lat_active=True, des_angle=0.0)
    self.controller.handle_angle_lateral(cc, cs)
    cs = _cs(angle=0.0, rate=0.0, torque=5.0, v=5.0)
    cc = _cc(lat_active=True, des_angle=20.0)
    self.controller.handle_angle_lateral(cc, cs)
    self.assertAlmostEqual(self.controller.apply_angle_last, 0.8, places=2)

  def test_highway_light_nudge_keeps_request(self):
    """Torque 60 is below the 80 highway bar — keep lane, do not yield."""
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 5.0, 25.0))
    for _ in range(8):
      self.controller.handle_angle_lateral(_cc(True, 2.0), _cs(1.0, 5.0, 60.0, 25.0))
    self.assertFalse(self.controller.angle_yielding)
    self.assertTrue(self.controller.lat_active_prev)

  def test_highway_single_spike_does_not_yield(self):
    """One 90-torque frame (road bump) must not drop Request."""
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 5.0, 25.0))
    self.controller.handle_angle_lateral(_cc(True, 2.0), _cs(1.0, 5.0, 90.0, 25.0))
    self.assertFalse(self.controller.angle_yielding)
    self.assertTrue(self.controller.lat_active_prev)

  def test_highway_manual_nudge_yields(self):
    """Sustained torque 90 for debounce frames drops Request."""
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 5.0, 25.0))
    self.assertTrue(self.controller.lat_active_prev)
    n = self.controller.p.LKAS_ANGLE_HWY_YIELD_DEBOUNCE
    for _ in range(n):
      self.controller.handle_angle_lateral(_cc(True, 2.0), _cs(1.0, 5.0, 90.0, 25.0))
    self.assertTrue(self.controller.angle_yielding)
    self.assertFalse(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 1.0)

  def test_highway_blinker_keeps_high_yield(self):
    """ALC / blinker on highway still uses the 120 hand threshold."""
    self.controller.handle_angle_lateral(
      _cc(True, 0.0), _cs(0.0, 0.0, 5.0, 25.0, left_blinker=True))
    for _ in range(6):
      self.controller.handle_angle_lateral(
        _cc(True, 2.0), _cs(1.0, 5.0, 90.0, 25.0, left_blinker=True))
    self.assertFalse(self.controller.angle_yielding)
    self.assertTrue(self.controller.lat_active_prev)

  def test_highway_manual_resume_is_slow(self):
    """After a highway no-blinker yield, 0.2 s calm must not snap Request back."""
    self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(0.0, 0.0, 5.0, 25.0))
    for _ in range(self.controller.p.LKAS_ANGLE_HWY_YIELD_DEBOUNCE):
      self.controller.handle_angle_lateral(_cc(True, 2.0), _cs(1.0, 26.0, 90.0, 25.0))
    self.assertTrue(self.controller.angle_yielding)

    early = (self.controller.p.LKAS_ANGLE_YIELD_MIN_FRAMES +
             self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES + 5)
    for _ in range(early):
      self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(1.0, 0.0, 5.0, 25.0))
    self.assertTrue(self.controller.angle_yielding)
    self.assertFalse(self.controller.lat_active_prev)

    rest = (self.controller.p.LKAS_ANGLE_RESUME_CALM_FRAMES_HWY + 5)
    for _ in range(rest):
      if not self.controller.angle_yielding:
        break
      self.controller.handle_angle_lateral(_cc(True, 0.0), _cs(1.0, 0.0, 5.0, 25.0))
    self.assertFalse(self.controller.angle_yielding)
    self.assertTrue(self.controller.lat_active_prev)
    self.assertAlmostEqual(self.controller.apply_angle_last, 1.0)


if __name__ == "__main__":
  unittest.main()
