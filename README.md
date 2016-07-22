## unzipped data files available at //nas/dataset/talkingdata larger than 1 GB

## File descriptions

### gender_age_train.csv, gender_age_test.csv - the training and test 

    group: this is the target variable you are going to predict

### events.csv, app_events.csv - when a user uses TalkingData SDK, the event gets logged in this data. Each event has an event id, location (lat/long), and the event corresponds to a list of apps in app_

    timestamp: when the user is using an app with TalkingData 

### app_labels.csv - apps and their labels, the label_id's can be used to join with label_categories

### label_categories.csv - apps' labels and their categories in text

### phone_brand_device_model.csv - device ids, brand, and models

    phone_brand: note that the brands are in Chinese (translation courtesy of user fromandto)
    
    - 三星 samsung
    - 天语 Ktouch
    - 海信 hisense
    - 联想 lenovo
    - 欧比 obi
    - 爱派尔 ipair
    - 努比亚 nubia
    - 优米 youmi
    - 朵唯 dowe
    - 黑米 heymi
    - 锤子 hammer
    - 酷比魔方 koobee
    - 美图 meitu
    - 尼比鲁 nibilu
    - 一加 oneplus
    - 优购 yougo
    - 诺基亚 nokia
    - 糖葫芦 candy
    - 中国移动 ccmc
    - 语信 yuxin
    - 基伍 kiwu
    - 青橙 greeno
    - 华硕 asus
    - 夏新 panosonic
    - 维图 weitu
    - 艾优尼 aiyouni
    - 摩托罗拉 moto
    - 乡米 xiangmi
    - 米奇 micky
    - 大可乐 bigcola
    - 沃普丰 wpf
    - 神舟 hasse
    - 摩乐 mole
    - 飞秒 fs
    - 米歌 mige
    - 富可视 fks
    - 德赛 desci
    - 梦米 mengmi
    - 乐视 lshi
    - 小杨树 smallt
    - 纽曼 newman
    - 邦华 banghua
    - E派 epai
    - 易派 epai
    - 普耐尔 pner
    - 欧新 ouxin
    - 西米 ximi
    - 海尔 haier
    - 波导 bodao
    - 糯米 nuomi
    - 唯米 weimi
    - 酷珀 kupo
    - 谷歌 google
    - 昂达 ada
    - 聆韵 lingyun

### sample_submission.csv - a sample submission file in the correct format

## Evaluation

Submissions are evaluated using the multi-class logarithmic loss. Each device has been labeled with one true class. For each device, you must submit a set of predicted probabilities (one for each class). The formula is then,

`logloss=−1N∑i=1N∑j=1Myijlog(pij)`,

where N is the number of devices in the test set, M is the number of class labels, log is the natural logarithm, yij is 1 if device i belongs to class j and 0 otherwise, and pij is the predicted probability that observation i belongs to class j.

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with `max(min(p,1−10−15),10−15)`.

## Submission File

You must submit a csv file with the device id, and a probability for each class.

The 12 classes to predict are:

    'F23-', 'F24-26','F27-28','F29-32', 'F33-42', 'F43+',
    'M22-', 'M23-26', 'M27-28', 'M29-31', 'M32-38', 'M39+'

The order of the rows does not matter. The file must have a header and should look like the following:

    device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+
    1234,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833
    5678,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833,0.0833
    ...
    
## Team Limits
    There is no maximum team size.
    Submission Limits

    You may submit a maximum of 5 entries per day.

    You may select up to 2 final submissions for judging.

## Competition Timeline
    Start Date: 7/11/2016 3:03:30 PM UTC
    Merger Deadline: 8/29/2016 11:59:00 PM UTC
    First Submission Deadline: 8/29/2016 11:59:00 PM UTC
    End Date: 9/5/2016 11:59:00 PM UTC
