import pytest
import numpy as np

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../src"))
from kalman import InterfaceObject, ObjectTracker


def test_ObjectTracker_tracks_two_objects():
    # Given
    measurements = [
        [
            InterfaceObject(
                state=np.array([1.0, 2.0]),
                covariance=np.array([1.0, 2.0]),
                time=2)
        ],
        [
            InterfaceObject(
                state=np.array([6.0, 1.0]),
                covariance=np.array([0.5, 0.5]),
                time=5),
            InterfaceObject(
                state=np.array([5.0, 2.0]),
                covariance=np.array([1.0, 1.0]),
                time=5)
        ]
    ]
    objTracker = ObjectTracker()

    # When
    for t in range(1, 10):
        objTracker.predict_objects(t)
        for measurement in measurements:
            if measurement[0].time == t:
                objTracker.process_measurement(measurement)

    # Then
    assert len(objTracker.objects) == 2
    tolerance = 0.05
    assert objTracker.objects[0].state[0] == pytest.approx(
        10.1, tolerance)
    assert objTracker.objects[0].state[1] == pytest.approx(
        0.95, tolerance)
    assert objTracker.objects[1].state[0] == pytest.approx(
        13, tolerance)
    assert objTracker.objects[1].state[1] == pytest.approx(
        2, tolerance)
