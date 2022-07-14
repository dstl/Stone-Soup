import pandas as pd
import glob

"""A Python script which will allow you to enter a directory of simulations and average the results.
This is typically already done automatically with the run manager however, may be needed for some
cases in AWS instances or HPC when runs are split over several nodes."""


def average_simulations(dataframes, length_files):
    """Takes list of dataframes and averages them on cell level

    Parameters
    ----------
    dataframes : Sequence[DataFrame]
        A set of pandas dataframes to be averaged
    length_files : int
        Length of dataframe set

    Returns
    -------
    DataFrame : pandas DataFrame
        Returns the average of dataframes loaded from CSV files.
    """
    timestamp = dataframes.iloc[:, -1]
    df = dataframes.iloc[:, :-1].div(length_files)
    df["timestamp"] = timestamp
    return df


def sum_simulations(directory, chunk_size: int):
    """Sums metrics.csv files and processes them in batches to reserve memory space.

    Parameters
    ----------
    directory : str
        directory path where metrics.csv is located
    chunk_size : int
        size of batches

    Returns
    -------
    DataFrame : pandas DataFrame
         Returns the sum of dataframes loaded from CSV files.
    """
    all_files = glob.glob(f'./{directory}*/run*[!_]/metrics.csv', recursive=True)
    batch = batch_list(all_files, chunk_size)
    summed_dataframe = pd.DataFrame()
    for files in batch:
        dfs = [pd.read_csv(file, infer_datetime_format=True) for file in files]
        averagedf = pd.concat(dfs, ignore_index=False).groupby(level=0).sum()
        averagedf["timestamp"] = dfs[0].iloc[:, -1]
        if not summed_dataframe.empty:
            summed_dataframe = summed_dataframe + averagedf
        if summed_dataframe.empty:
            summed_dataframe = averagedf

    return summed_dataframe, len(all_files)


def batch_list(lst, n):
    """ Splits list into batches/chunks to be used when memory issues arise with
    averaging large datasets.

    Parameters
    ----------
    lst : list
        List object
    n : int
        number of items per batch

    Returns
    -------
    List
        Returns a list containing n number of batches
    """
    if n == 0:
        n = len(lst)
    chunked_lists = []
    for i in range(0, len(lst), n):
        chunked_lists.append(lst[i:i + n])
    return chunked_lists


def average_metrics(batch_size=200):
    """Handles the averaging of the metric files for both single simulations
    and multi simulations.

    In future updates better memory handling should be implemented to automatically
    load as many dataframes as possible into memory to provide a more efficient process.

    Parameters
    ----------
    batch_size : int
        Size of the batches to split the dataframes. Default is 200. This may need adjusting for
        very large datasets to save memory space.
    """

    config_filename = input("Enter the configuration filename \
                            (testing.yaml_2022_03_16 for example): ")

    try:
        directory = glob.glob(f'./{config_filename}*/simulation*', recursive=False)
        if directory:
            for simulation in directory:
                summed_df, sim_amt = sum_simulations(simulation, batch_size)
                df = average_simulations(summed_df, sim_amt)
                df.to_csv(f"./{simulation}/average.csv", index=False)
        else:
            directory = glob.glob(f'{config_filename}*', recursive=False)
            summed_df, sim_amt = sum_simulations(directory, batch_size)
            df = average_simulations(summed_df, sim_amt)
            df.to_csv(f"./{config_filename}*/average.csv", index=False)
        print(f"{config_filename} files have been Averaged")
    except Exception as e:
        print(f"Failed to average simulations. {e}")


average_metrics()
