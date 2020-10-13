
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import copy


class Association():
    def __init__(self, obj_measurement, m_idx, obj, o_idx):
        self.distance = abs(
            obj_measurement.state[0] - obj.state[0])
        self.measurement_idx = m_idx
        self.object_idx = o_idx


class AssociationContainer():
    def __init__(self):
        self.associations = []
        self.gating_distance = 2

    def _create_possible_assos(self, measurement, objects):
        for m_idx, obj_measurement in enumerate(measurement):
            for o_idx, obj in enumerate(objects):
                asso = Association(obj_measurement, m_idx, obj, o_idx)
                if asso.distance < self.gating_distance:
                    self.associations.append(asso)

    def _assos_share_obj_or_measurement(self, asso1, asso2):
        if asso1.measurement_idx == asso2.measurement_idx:
            return True
        elif asso1.object_idx == asso2.object_idx:
            return True
        else:
            return False

    def _leave_max_one_measurement_per_object(self):
        self.associations.sort(key=lambda x: x.distance)
        for i, asso in enumerate(self.associations):
            for j in range(i + 1, len(self.associations)):
                if self._assos_share_obj_or_measurement(
                        asso, self.associations[j]):
                    self.associations.pop(j)

    def associate_measurement_to_objects(self, measurement, objects):
        self._create_possible_assos(measurement, objects)
        self._leave_max_one_measurement_per_object()

    def get_obj_idx(self, measurement_idx):
        obj_idx = None
        for asso in self.associations:
            if asso.measurement_idx == measurement_idx:
                obj_idx = asso.object_idx
                break
        return obj_idx


class ObjectTracker():
    def __init__(self):
        self.objects = []
        self.id_of_next_added_object = 1

    def predict_objects(self, target_time):
        for obj in self.objects:
            obj.predict_object(target_time)

    def _add_object(self, measured_obj):
        self.objects.append(
            KalmanTrackedObject.create_obj(
                self.id_of_next_added_object, measured_obj))
        self.id_of_next_added_object += 1

    def _process_measurement_data(self, measurement):
        assoCont = AssociationContainer()
        assoCont.associate_measurement_to_objects(measurement, self.objects)
        for idx, measured_obj in enumerate(measurement):
            obj_idx = assoCont.get_obj_idx(idx)
            if obj_idx is None:
                # Measurement was not associated, create a new object
                self._add_object(measured_obj)
            else:
                self.objects[obj_idx].update(measured_obj)

    def process_measurement(self, measurement):
        self.predict_objects(measurement[0].time)
        self._process_measurement_data(measurement)

    def print_objects(self):
        for obj in self.objects:
            print(obj)


class InterfaceObject():
    def __init__(self, state, covariance, time):
        # State vector
        # position
        # velocity
        self.state = state
        if covariance.shape == (2,):
            self.covariance = np.array([[covariance[0], 0.0],
                                        [0.0, covariance[1]]])
        else:
            self.covariance = covariance
        self.time = time


class KalmanTrackedObject(InterfaceObject):
    def __init__(self, id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_per_time = np.array([[0.2, 0.1]])
        self.id = id

    def update(self, measured_obj):
        kalman_gain = self.covariance.dot(
            inv(self.covariance + measured_obj.covariance))
        self.state = self.state + kalman_gain.dot(
            measured_obj.state - self.state)
        self.covariance = (
            np.identity(self.covariance.shape[0]) - kalman_gain).dot(
                self.covariance)

    def predict_object(self, target_time):
        delta_t = target_time - self.time
        pred_matrix = np.array([[1.0, delta_t],
                                [0.0, 1.0]])
        self.state = pred_matrix.dot(self.state)
        self.covariance = pred_matrix.transpose().dot(
            self.covariance).dot(pred_matrix)
        self.covariance += np.identity(
            self.covariance.shape[0]) * self.noise_per_time * delta_t
        self.time = target_time

    def __str__(self):
        output = f"""Time: {self.time}
Object Id: {self.id}
State: {self.state}
Covariance:\n{self.covariance}
"""
        return output

    @classmethod
    def create_obj(cls, id, interfaceObject):
        return cls(
            id,
            interfaceObject.state,
            interfaceObject.covariance,
            interfaceObject.time)


def get_tracks(object_lists):
    object_lists = [o for o in object_lists if o]
    times = [obj_list[0].time for obj_list in object_lists]
    tracks = []
    max_id = max(
        [max([obj.id for obj in obj_list]) for obj_list in object_lists])
    for id in range(1, max_id + 1):
        track = {
            "time": [],
            "position": [],
            "covariance": []}
        for idx, t in enumerate(times):
            for obj in object_lists[idx]:
                if obj.id == id:
                    track["time"].append(t)
                    track["position"].append(obj.state[0])
                    track["covariance"].append(obj.covariance[0, 0])
                    break
        tracks.append(track)
    return tracks


def print_tracks(object_lists):
    tracks = get_tracks(object_lists)
    for track in tracks:
        plt.errorbar(
            track["time"], track["position"],
            yerr=track["covariance"], fmt='-o')
    plt.title('Kalman tracked objects')
    plt.ylabel('Lateral position [m]')
    plt.xlabel('Time [s]')
    plt.show()


if __name__ == "__main__":
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
    object_lists = []
    for t in range(1, 10):
        objTracker.predict_objects(t)
        for measurement in measurements:
            if measurement[0].time == t:
                objTracker.process_measurement(measurement)
        objTracker.print_objects()
        object_lists.append(copy.deepcopy(objTracker.objects))

    print_tracks(object_lists)
