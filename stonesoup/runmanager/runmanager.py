import logging
import sys
from .runmanagercore import RunManagerCore

if __name__ == "__main__":
    args = sys.argv[1:]

    try:
        configInput = args[0]

    except Exception as e:
        configInput = ""
        logging.error(e)

    try:
        parametersInput = args[1]
    except Exception as e:
        parametersInput = ""
        logging.error(e)
        # parametersInput= "C:\\Users\\gbellant\\Documents\\Projects\\Serapis\\dummy3.json"

    try:
        groundtruthSettings = args[2]
    except Exception as e:
        groundtruthSettings = 1
        logging.error(e)

    rmc = RunManagerCore()

    rmc.run(configInput, parametersInput, groundtruthSettings)
