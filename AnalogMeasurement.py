import numpy as np

from saleae.range_measurements import AnalogMeasurer


class RiseFallTime(AnalogMeasurer):
    supported_measurements = ["v_30p", "v_70p", "t_rise", "t_fall"]

    # Initialize your measurement extension here
    # Each measurement object will only be used once, so feel free to do all per-measurement initialization here
    def __init__(self, requested_measurements):
        super().__init__(requested_measurements)
        self.samples = []
        self.sampling_period = None

        if "v_30p" in self.supported_measurements:
            self.measure_v_30p = True
        if "v_70p" in self.supported_measurements:
            self.measure_v_70p = True
        if "t_rise" in self.supported_measurements:
            self.measure_t_rise = True
        if "t_fall" in self.supported_measurements:
            self.measure_t_fall = True


    # This method will be called one or more times per measurement with batches of data
    # data has the following interface
    #   * Iterate over to get Voltage values, one per sample
    #   * `data.samples` is a numpy array of float32 voltages, one for each sample
    #   * `data.sample_count` is the number of samples (same value as `len(data.samples)` but more efficient if you don't need a numpy array)
    def process_data(self, data):
        self.samples.append(data.samples)
        if self.sampling_period is None:
            self.sampling_period = ((data.end_time - data.start_time) / data.sample_count).__float__()


    # This method is called after all the relevant data has been passed to `process_data`
    # It returns a dictionary of the request_measurements values
    def measure(self):
        data = np.concatenate(self.samples)

        max = np.max(data)

        v_30p = max * 0.3
        v_70p = max * 0.7

        t_rise = []
        t_fall = []

        if self.measure_t_rise or self.measure_t_fall:
            i_30p = None
            i_70p = None

            v = data[0]
            for i, n in enumerate(data[1:]):
                if (v >= v_30p and n < v_30p) or (v <= v_30p and n > v_30p):
                    i_30p = i
                    if i_70p is not None:
                        t_fall.append(i_30p - i_70p)
                        i_30p = i_70p = None
                if (v >= v_70p and n < v_70p) or (v <= v_70p and n > v_70p):
                    i_70p = i
                    if i_30p is not None:
                        t_rise.append(i_70p - i_30p)
                        i_30p = i_70p = None
                v = n        

        values = {}

        if self.measure_v_30p:
            values["v_30p"] = v_30p

        if self.measure_v_70p:
            values["v_70p"] = v_70p

        if self.measure_t_rise and len(t_rise):
            values["t_rise"] = np.average(t_rise) * self.sampling_period
        
        if self.measure_t_fall and len(t_fall):
            values["t_fall"] = np.average(t_fall) * self.sampling_period

        return values
