def initTemp(temp):
    temp["current_epoch"] = 0
    temp["scheduler_last_epoch"] = 0
    temp["min_save_metric"] = 1e10
    temp["max_save_metric"] = 0

