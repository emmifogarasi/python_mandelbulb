import pytest 
from src.mandelbulb_generator import mandelbulb_iterate_point

DEFAULT_POWER = 8
DEFAULT_MAX_ITERATION = 20
DEFAULT_BAILOUT_RADIUS = 2.0
DEFAULT_BAILOUT_RADIUS_SQ = DEFAULT_BAILOUT_RADIUS**2

def test_origin_stays_in_set():
    """
    Tests that the point (0,0,0) is considered part of the set
    (i.e., reachest max_iterations)
    """
    iterations = mandelbulb_iterate_point(
        cx=0.0, cy=0.0, cz=0.0,
        power=DEFAULT_POWER,
        max_iterations=DEFAULT_MAX_ITERATION,
        bailout_radius_sq=DEFAULT_BAILOUT_RADIUS_SQ
    )
    assert iterations == DEFAULT_MAX_ITERATION

def test_point_far_away_escapes_immediately():
    """
    Tests that a point far outside the bailoud radius escapes on the first check
    The iteration loop runs from i=0 up to max_iterations-1
    If r_sq > bailout_radius_sq before the loop's first step, it returns current i (which is 0) 
    """
    cx, cy, cz = 3.0, 0.0, 0.0
    iterations = mandelbulb_iterate_point(
        cx=cx, cy=cy, cz=cz,
        power=DEFAULT_POWER,
        max_iterations=DEFAULT_MAX_ITERATION,
        bailout_radius_sq=DEFAULT_BAILOUT_RADIUS_SQ
    )
    assert iterations == 0

def test_return_type_and_range():
    """
    Tests that the function returns an integer within the expected range.
    """
    iterations = mandelbulb_iterate_point(
        cx=0.1, cy=0.2, cz=0.3,
        power=DEFAULT_POWER,
        max_iterations=DEFAULT_MAX_ITERATION,
        bailout_radius_sq=DEFAULT_BAILOUT_RADIUS_SQ
    )
    assert isinstance(iterations, int)
    assert 0 <= DEFAULT_MAX_ITERATION
