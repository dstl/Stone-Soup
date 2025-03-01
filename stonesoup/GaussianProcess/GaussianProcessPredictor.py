class GaussianProcessPredictor:
    def __init__(self, gp):
        """
        Initialize the Gaussian Process Predictor.

        Parameters
        ----------
        gp : GaussianProcess
            A trained Gaussian Process model.
        """
        self.gp = gp

    def predict(self, timestamps, selected_dim="x"):
        """
        Predict for given timestamps and selected dimensions.

        Parameters
        ----------
        timestamps : list of datetime
            Timestamps to predict for.
        selected_dim : str
            "x" for dimension 0, "y" for dimension 1, or "x,y" for both dimensions.

        Returns
        -------
        dict
            A dictionary where keys are dimensions and values are the posterior means and covariances.
        """
        if not self.gp.is_trained:
            raise RuntimeError("GaussianProcess must be trained before prediction.")

        # Map selected_dim to dimension indices
        if selected_dim == "x":
            dims = [0]
        elif selected_dim == "y":
            dims = [1]
        elif selected_dim == "x,y":
            dims = list(range(self.gp.dimensions))
        else:
            raise ValueError("Invalid selected_dim. Use 'x', 'y', or 'x,y'.")

        # Get predictions from GaussianProcess
        predictions = self.gp.posterior(timestamps)

        # Filter results based on selected dimensions
        filtered_results = {dim: predictions[dim] for dim in dims}
        return filtered_results
