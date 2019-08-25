class Plot:
    data_points = {}

    def __init__(self, data_points):
        self.data_points = data_points

    def draw(self, width, height, title):
        plt.figure(figsize=(width, height))
        for data_point in self.data_points.iterrows():
            plt.plot(data_point['data'][1], label=data_point['label'][1], marker=data_point['marker'][1],
                     color=data_point['color'][1], linestyle=data_point['linestyle'][1])

        plt.title(title)
        plt.legend()
