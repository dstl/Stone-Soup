


def test_pda(UpdaterClass, measurement_model, prediction, measurement):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[8]]))
    detections = {detection1, detection2}

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9, prob_gate=0.99)

    mulltihypothesis = \
        hypothesiser.hypothesise(track, detections, timestamp)

    data_associator = PDA(hypothesiser=hypothesiser)

    hypotheses = data_associator.associate({track}, detections,
                                           start_time + timedelta(seconds=1))

    posterior_state = pdaupdater.update(hypotheses, gm_method=True)


