from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CustomCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # episode.user_data = {"final": {}, "running": {}}

        episode.user_data = {
            metric: []
            for metric in [
                "cumulative_distance",
                "basic_metric",
                "OSPA_distances",
                "detected_this_step",
                "agent_outside",
            ]
        }

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):

        agent_with_info = [
            episode.last_info_for(i)
            for i in episode.get_agents()
            if episode.last_info_for(i) != {}
        ]

        for custom_metric, value in episode.user_data.items():
            if agent_with_info:
                data = agent_with_info[0]
                episode.user_data[custom_metric].append(data[custom_metric])

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for key, value in episode.user_data.items():
            if key in [
                "cumulative_distance",
                "basic_metric",
                "OSPA_distances",
                "detected_this_step",
                "agent_outside",
            ]:
                # Take average of formation reward per step per agent per episode
                episode.custom_metrics[key] = sum(value) / len(value)
            else:
                # Max of episode values for all others, need the largest number
                episode.custom_metrics[key] = max(value)

            episode.hist_data[key] = value
