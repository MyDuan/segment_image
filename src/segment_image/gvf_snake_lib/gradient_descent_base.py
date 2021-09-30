FLOAT_MAX = 3.40282e+38


class GradientDecentBase:
    def __init__(self, step_size):
        self.step_size = step_size
        self.last_energy_ = FLOAT_MAX

    def run(self, max_iteration):
        self.initialize()
        self.last_energy_ = self.compute_energy()
        current_iter = 0
        while not self.is_terminate(current_iter, max_iteration):
            print("start update: " + str(current_iter))
            current_iter += 1
            self.update()
            print("finish update: " + str(current_iter))
            new_energy = self.compute_energy()
            print("new_energy" + str(new_energy))
            print("====================")
            if new_energy < self.last_energy_:
                self.last_energy_ = new_energy

    def is_terminate(self, current_iter, max_iteration):
        return current_iter >= max_iteration
