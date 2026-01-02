class DistanceSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.tracks = {}

    def update(self, track_id, new_distance):
        if new_distance is None:
            return None
        
        if track_id not in self.tracks:
            self.tracks[track_id] = new_distance
        else:
            prev_distance = self.tracks[track_id]
            self.tracks[track_id] = (self.alpha * new_distance) + ((1 - self.alpha) * prev_distance)
        
        return self.tracks[track_id]
    