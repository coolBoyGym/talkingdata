data_app_events = '../data/app_events.csv'
data_app_labels = '../data/app_labels.csv'
data_events = '../data/events.csv'
data_gender_age_test = '../data/gender_age_test.csv'
data_gender_age_train = '../data/gender_age_train.csv'
data_label_categories = '../data/label_categories.csv'
data_phone_brand_device_model = '../data/phone_brand_device_model.csv'
data_sample_submission = '../data/sample_submission.csv'

if __name__ == '__main__':
    for n in [data_app_events, data_app_labels, data_events, data_gender_age_test, data_gender_age_train,
              data_label_categories, data_phone_brand_device_model, data_sample_submission]:
        with open(n, 'r') as fin:
            print next(fin)

        print '\n\n\n\n'
