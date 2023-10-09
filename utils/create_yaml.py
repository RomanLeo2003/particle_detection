import yaml

particle_yaml = dict(
    train=r"C:\Users\user\PycharmProjects\particle_detection\labels\train",
    val=r"C:\Users\user\PycharmProjects\particle_detection\labels\val",
    test=r"C:\Users\user\PycharmProjects\particle_detection\labels\test",
    nc=1,
    names =['particle']
)

tracker_yaml = dict(
    tracker_type='botsort',  # tracker type, ['botsort', 'bytetrack']
    track_high_thresh=0.25,  # threshold for the first association
    track_low_thresh=0.07,  # threshold for the second association
    new_track_thresh=0.7,  # threshold for init new track if the detection does not match any tracks
    track_buffer=30,  # buffer to calculate the time when to remove tracks
    match_thresh=0.25,
    cmc_method='sparseOptFlow',  # method of global motion compensation
    # ReID model related thresh (not supported yet)
    proximity_thresh=0.35,
    appearance_thresh=0.25,
    with_reid=True
)

with open('particles.yaml', 'w') as outfile:
    yaml.dump(particle_yaml, outfile, default_flow_style=True)

with open('tracker.yaml', 'w') as outfile:
    yaml.dump(tracker_yaml, outfile, default_flow_style=True)