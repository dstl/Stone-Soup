from stonesoup.wrapper.matlab import MatlabWrapper

engine = MatlabWrapper.connect_engine('my_matlab3')

wrapper = MatlabWrapper(None, matlab_engine=engine)

print(wrapper.matlab_engine.version())
