import os
import time
from codecarbon import EmissionsTracker
from src.green_computing.singleton_meta_definition import SingletonMeta

class EmissionTrackerManager(metaclass=SingletonMeta):
    
    def __init__(self, saving_path):
        self.full_path = os.path.join(saving_path, 
                                      "codecarbon",
                                      time.strftime("%Y-%m-%d", time.localtime()))
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)

        self.is_tracking = False
        self.tracker = None
        self.current_function = None
        self.results_string = "CO2_emissions_kg"

    def start_tracking(self, function_name):
        if self.is_tracking:
            raise RuntimeError("A tracker is already active. Stop it before starting another.")
        
        self.current_function = function_name

        self.tracker = EmissionsTracker(
            output_dir=self.full_path,
            output_file=f'{self.current_function}_emissions.csv',
            save_to_api=False,
            measure_power_secs=20,
            log_level = "WARNING"
        )
        self.tracker.start()
        self.is_tracking = True

        return self.tracker

    def stop_tracking(self, results):
        if not self.is_tracking:
            raise RuntimeError("No tracker is currently active.")
        
        emissions = self.tracker.stop()

        try:
            results[self.results_string][self.current_function] = emissions
        except Exception as e:
            print(f"Codecarbon encountered an error, but the execution is continuing. Error details: {e}")

        self.is_tracking = False
        self.tracker = None
        self.current_function = None

        return emissions
    