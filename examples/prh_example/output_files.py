import numpy as np


def output_tracks(filename, start_time, all_tracks):
    # Output all the tracks from the trackers
    f = open(filename, "w")
    f.write(str(start_time) + "\n") # start time
    f.write(str(len(all_tracks)) + "\n") # number of hierarchy levels
    for i_level, level in enumerate(all_tracks):
        f.write(str(i_level) + "\n") # index of this level
        f.write(str(len(level)) + "\n") # number of trackers at this level
        for i_tracker, tracker in enumerate(level):
            f.write(str(i_tracker) + "\n") # index of this tracker
            f.write(str(len(tracker)) + "\n") # number of tracks
            for i_track, track in enumerate(tracker):
                f.write(str(len(track)) + "\n") # number of states in track
                for x in track:
                    f.write(str(x.timestamp) + "\n") # state timestamp
                    f.write(str(x.state_vector.flatten()) + "\n") # state value
                    f.write(str(x.covar.flatten()) + "\n") # covariance
    f.close()


def output_meas(filename, start_time, platform_positions, all_detections):
    np.set_printoptions(linewidth=np.inf)
    f = open(filename, "w")
    f.write(str(start_time) + "\n")
    f.write(str(len(platform_positions)) + "\n")
    for pos in platform_positions:
        f.write(str(pos.flatten()) + "\n")
    f.write(str(len(all_detections)) + "\n")
    for i_level, level_det in enumerate(all_detections):
        f.write(str(i_level) + "\n")
        f.write(str(len(level_det)) + "\n")
        for i_node, node_det in enumerate(level_det):
            f.write(str(i_node) + "\n")
            f.write(str(len(node_det)) + "\n")
            for det in node_det:
                f.write(str(det.timestamp) + "\n")
                f.write(str(np.array(det.state_vector).flatten()) + "\n")
                f.write(str(np.array(det.measurement_model.noise_covar).flatten()) + "\n")
                if i_level > 0:
                    f.write(str(np.array(det.measurement_model.h_matrix.shape)) + "\n")
                    f.write(str(np.array(det.measurement_model.h_matrix).flatten()) + "\n")
    f.close()


def output_truth(filename, start_time, truth):
    # Output the truth
    f = open(filename, "w")
    f.write(str(start_time) + "\n")
    f.write(str(len(truth)) + "\n")
    for target in truth:
        f.write(str(len(target.states)) + "\n")
        for x in target.states:
            f.write(str(x.timestamp) + "\n")
            f.write(str(x.state_vector.flatten()) + "\n")
    f.close()
