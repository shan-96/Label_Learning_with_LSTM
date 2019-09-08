from model.CommentSnorkel import CommentSnorkel

# from model.PrepareData import PrepareData
# from scrap.GlobalVars import DATAFILENAME

if __name__ == "__main__":
    # prepare = PrepareData(DATAFILENAME)
    # prepare.prepare_data()
    model_trainer = CommentSnorkel()
    trainer1, Y_Fits = model_trainer.getTrainedModel1()
    trainer2 = model_trainer.getTrainedModel2()
