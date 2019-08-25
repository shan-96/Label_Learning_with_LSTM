class Plot:
    data_points = []

    def __init__(self, data_points):
        self.data_points = data_points

    def draw(self, width, height):
        plt.figure(figsize=(width, height))
        for data_point in self.data_points:
            plt.plot(data_point, label=data_point)
            plt.plot(f1, color='black', marker='o', linestyle='dashed', label='F1')
            plt.plot(acc, color='blue', marker='*', linestyle='dashed', label='Acc')
            plt.plot(recall, color='yellow', marker='o', linestyle='dashed', label='Recall')
            plt.plot(prec, color='red', marker='*', linestyle='dashed', label='Precision')

        plt.title("--Various Accuracy Measures--")
        plt.legend()
