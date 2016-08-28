# Features

### feature_name, feature_type, spaces, rank

    phone_brand, one_hot, 131, 1
    device_model, one_hot, 1666, 1
    device_long_lat, multi, 10, 10
        mean, max, min, std, median ...
    _norm, multi, 10, 10
        scale long/lat to [0, 1]
    device_event_num, num, 1, 1
    _norm, ...
    device_day_event_num, multi, 31, 8
    _norm, ...
    device_hour_event_num, multi, 24, 24
    _norm, ...
    device_day_hour_event_num, multi, 744, 169
    _norm, ...

    installed_app, multi, 19237, 3446
    _freq, ...
    active_app, multi, 19237, 1342
    _freq, ...

    installed_app_label, multi, 507, 411
    _freq, ...
    active_app_label, multi, 507, 357
    _freq, ...

    time_label_group:


### concat features

    concat_1, [fea_phone_brand, fea_device_model, ]

    concat_2, [fea_phone_brand, fea_device_model,fea_device_long_lat, ]

    concat_3, [fea_phone_brand,fea_device_model,fea_device_long_lat,fea_device_event_num,fea_device_day_event_num,fea_device_hour_event_num,fea_device_day_hour_event_num, ]

    concat_4, [fea_phone_brand,fea_device_model,fea_device_long_lat,fea_device_event_num,fea_device_day_event_num,fea_device_hour_event_num,fea_device_day_hour_event_num,fea_installed_app,fea_active_app, ]

    concat_5, [fea_phone_brand,fea_device_model,fea_device_long_lat,fea_device_event_num,fea_device_day_event_num,fea_device_hour_event_num,fea_device_day_hour_event_num,fea_installed_app,fea_active_app,fea_installed_app_label,fea_active_app_label, ]

    concat_2_norm, [fea_phone_brand,fea_device_model,fea_device_long_lat_norm, ]

    concat_3_norm, [fea_phone_brand,fea_device_model,fea_device_long_lat_norm,fea_device_event_num_norm,fea_device_day_event_num_norm,fea_device_hour_event_num_norm,fea_device_day_hour_event_num_norm])

    concat_4_norm, [fea_phone_brand,fea_device_model,fea_device_long_lat_norm,fea_device_event_num_norm,fea_device_day_event_num_norm,fea_device_hour_event_num_norm,fea_device_day_hour_event_num_norm,fea_installed_app_freq,fea_active_app_freq, ]

    concat_5_norm, [fea_phone_brand,fea_device_model,fea_device_long_lat_norm,fea_device_event_num_norm,fea_device_day_event_num_norm,fea_device_hour_event_num_norm,fea_device_day_hour_event_num_norm,fea_installed_app_freq,fea_active_app_freq,fea_installed_app_label_freq,fea_active_app_label_freq, ]

    concat_6, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label]

    concat_7, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_long_lat]

    concat_7_norm, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_long_lat_norm]

    concat_8, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_event_num,fea_device_weekday_event_num,fea_device_day_event_num,fea_device_hour_event_num, ])

    concat_8_norm, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_event_num_norm,fea_device_weekday_event_num_norm,fea_device_day_event_num_norm,fea_device_hour_event_num_norm, ]

    concat_9, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_long_lat,fea_device_event_num,fea_device_weekday_event_num,fea_device_day_event_num,fea_device_hour_event_num, ]

    concat_9_norm, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_long_lat_norm,fea_device_event_num_norm,fea_device_weekday_event_num_norm,fea_device_day_event_num_norm,fea_device_hour_event_num_norm, ])

    concat_10, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_hour_event_num_freq,fea_device_weekday_event_num_freq])

    concat_11, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_category])

    concat_12, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_diff_hour_category])

    concat_13, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_each_hour_category])

    concat_14, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label,fea_device_hour_event_num_freq,fea_device_weekday_event_num_freq,fea_device_long_lat_norm,])

    concat_15, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_category_num_tfidf])

    concat_16, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_cluster_40])

    concat_16_tfidf, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_cluster_40_num_tfidf])

    concat_16_2, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_cluster_100])

    concat_16_3, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_cluster_270])

    concat_17, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app_label_num_tfidf])

    concat_6_tfidf, [fea_phone_brand,fea_device_model,fea_installed_app_tfidf,fea_installed_app_label_tfidf])

    concat_20, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_active_app,fea_active_app_label])

    concat_21, [fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label,fea_device_day_event_num_freq,fea_device_hour_event_num_freq,fea_device_weekday_event_num_freq,fea_device_long_lat_norm,fea_active_app_label_category,fea_active_app_label_cluster_40]

    concat_22_8, [fea_phone_brand,fea_device_model,fea_installed_app_w2v_8,fea_installed_app_label_w2v_8]
    
    concat_22_16, [fea_phone_brand,fea_device_model,fea_installed_app_w2v_16,fea_installed_app_label_w2v_16]
    
    concat_22_32, [fea_phone_brand,fea_device_model,fea_installed_app_w2v_32,fea_installed_app_label_w2v_32]
    
    concat_22_64, [fea_phone_brand,fea_device_model,fea_installed_app_w2v_64,fea_installed_app_label_w2v_64]
    
    concat_22_128, [fea_phone_brand,fea_device_model,fea_installed_app_w2v_128,fea_installed_app_label_w2v_128]
    
    concat_6_ooee_64,[fea_phone_brand,fea_device_model,fea_installed_app_embedding,fea_installed_app_label_embedding])

### ensemble features

    emsemble_1, multi, 42215, 5890

    ensemble_2

    ensemble_3: [concat_1_gbtree_1, concat_1_gblinear_1, concat_6]

    ensemble_4, [fea_concat_1_gbtree_1,fea_concat_1_gblinear_1,fea_concat_2_gbtree_1,fea_concat_2_gblinear_1,fea_concat_2_norm_gbtree_1,fea_concat_2_norm_gblinear_1,fea_concat_3_gbtree_1,fea_concat_3_gblinear_1,fea_concat_3_norm_gbtree_1,fea_concat_3_norm_gblinear_1,fea_concat_4_gbtree_1,fea_concat_4_gblinear_1,fea_concat_4_norm_gbtree_1,fea_concat_4_norm_gblinear_1,fea_concat_5_gbtree_1,fea_concat_5_gblinear_1,# fea_concat_5_norm_gbtree_1,# fea_concat_5_norm_gblinear_1,fea_concat_6_gbtree_1,fea_concat_6])
    
    ensemble_5, [fea_concat_1_gblinear_1,fea_concat_1_gbtree_1,
                    fea_concat_2_norm_gblinear_1,fea_concat_2_norm_gbtree_1,
                    fea_concat_3_norm_gblinear_1,fea_concat_3_norm_gbtree_1,
                    fea_concat_4_gblinear_1,fea_concat_4_gbtree_1,fea_concat_4_norm_gblinear_1,fea_concat_4_norm_gbtree_1,
                    fea_concat_5_norm_gblinear_1,fea_concat_5_norm_gbtree_1,
                    fea_concat_6_embedding_64_mlp_for_ensemble,fea_concat_6_gbtree_1,fea_concat_6_mlp_136,fea_concat_6_mlp_for_ensemble,fea_concat_6_tfidf_gbtree_1,
                    fea_concat_7_norm_for_ensemble,
                    fea_concat_15_mlp_6,
                    fea_concat_16_mlp_3,
                    fea_concat_20_mlp_for_ensemble,
                    fea_concat_21_mlp_for_ensemble]
                    
    ensemble_6, [fea_ensemble_5,fea_phone_brand,fea_device_model,fea_installed_app,fea_installed_app_label]
    
    ensemble_7, [fea_ensemble_5,fea_concat_6]
    
# Models

### GBLinear

### GBTree

### LR

### FM

### MLP

### MNN

### CNN

### textCNN

# Ensemble Models

### Blending

### Bagging

# Submissions

## 16/8/12 Fri.

### gym
        concat_5, gbtree, {max_depth=3, eta=0.1, subsample=0.8, colsample=0.5, round=860, rate=0.2},
                                2.2824732, 2.27017, concat_5_gbtree_1
        concat_5, gblinear, {alpha=0.1, lambda=66, early_stop_round=1, round=3, rate=0.2},
                                2.29340173483, not yet, concat_5_gblinear_1
        concat_4, gbtree, {max_depth=3, eta=0.1, subsample=0.8, colsample=0.9, round=950, rate=0.2},
                                2.28721, 2.27817, concat_4_gbtree_1
        concat_4, gblinear, {alpha=0.05, lambda=44, early_stop_round=1, round=2, rate=0.2},
                                2.29869730537, not yet, concat_4_gblinear_1
        concat_4_norm, gbtree, {max_depth=3, eta=0.1, subsample=0.8, colsample=0.7, round=777, rate=0.2},
                                2.29613, not yet, concat_4_norm_gbtree_1
        concat_4_norm, gblinear, {alpha=0.1, lambda=13, early_stop_round=1, rate=0.2},
                                2.32331138572, not yet, concat_4_norm_gblinear_1

### rocky
        concat_1, gbtree            max_depth 7 eta 0.07 subsample 0.8 colsample_bytree 0.5 2.35875357389 2.39043890146
        concat_1, gblinear          alpha 0 lambda 7.5  2.3499658672 2.38944255486
        concat_2, gbtree            max_depth 3 eta 0.07 subsample 0.8 colsample_bytree 0.6 2.33399659152 2.38934798614
        concat_2, gblinear          alpha 0.001 lambda 8 2.34606485552 2.38613098915
        concat_2_norm, gbtree       max_depth 3 eta 0.07 subsample 0.8 colsample_bytree 0.6 2.33028431943 2.38669627018
        concat_2_norm, gblinear     alpha 0.001 lambda 7.5 2.34475563006 2.38551132474

### xepa
        concat_5_norm, gbtree, {max_depth=4, eta=0.1, subsample=0.7, colsample=0.7}, [556]   train-mlogloss:1.89167	eval-mlogloss:2.28766, concat_5_norm_gbtree_1
        concat_5_norm, gblinear, {0.001, 10}, [2]	train-mlogloss:2.15605	eval-mlogloss:2.32268, concat_5_norm_gblinear_1
        concat_5_norm, lr, {0, 0}

## 2016/8/13 Sat.

### xepa:
        ensemble, average, {[ 0.01968983,  0.0391137 ,  0.01906632,  0.00886792,  0.0204471 ,
                                0.02016268,  0.25404595,  0.2369304 ,  0.22133675,  0.16033936,]}, 2.03700311,  2.28009707, average_1.log

### gym:
        ensemble_2, gbtree, {max_depth=2, eta=0.01, subsample=0.01, colsample=0.01, early_stop_round=50,
                                round=3980, lambda=19, alpha=0.1, rate=0.2}
                            valid_score=2.27704846522,  real_score=2.27079, ensemble_2_gbtree

        concat_1, factorization_machine, {learning_rate=0.15, l1_w=0.2, l1_v=0.1, num_round=300, batch_size=15,
                                early_stopping_round=10, } valid_score=2.35431 real_score=not yet

        concat_6, gbtree, {max_depth=3, eta=0.1, subsample=0.7, colsample=0.8, early_stop_round=25,
                                round=720, lambda=1, alpha=0.1, rate=0.2}
                             valid_score=2.27860747937, real_score=2.26798, concat_6_gbtree

        concat_7, gbtree, {max_depth=3, eta=0.1, subsample=0.8, colsample=0.9, early_stop_round=25,
                                round=673, lambda=1, alpha=0.1, rate=0.2}
                             valid_score=2.2813284161, real_score=not yet, concat_7_gbtree

        concat_10, gbtree, {max_depth=3, eta=0.1, subsample=0.6, colsample=0.8, early_stop_round=30,
                                round=512, lambda=1, alpha=0.1, rate=0.2}
                             valid_score=2.28447753216, real_score=not yet, concat_10_gbtree


## 2016/8/14 Sun.

## 2016/8/15 Mon.

### xepa:
        concat_6_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 0.5]
            learning_rate = 0.2
            num_round = 470
            471	2.205019	2.205045	2.258668
            score: 2.24483

### rocky:
        concat_6_mlp:
            opt_algo:
            drops:
            multi-layer:

### gym:
        mlp on other features

        concat_6_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 0.5]
            learning_rate = 0.2
            num_round = 500
            457	2.207887	2.207884	2.258564
            score:not yet

        concat_10_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 0.5]
            learning_rate = 0.2
            [475]	train_score: 2.202560	valid_score: 2.258602
            score:not yet

## 2016/8/16:

### rocky:
        concat_6
        (100,10) lr=0.5
        [137]	train_score: 2.369727	valid_score: 2.365398
        (100,20) lr=0.5
        [141]	train_score: 2.362218	valid_score: 2.358099
        (100,50) lr=0.5
        [181]	train_score: 2.322096	valid_score: 2.328415
        (100,100) lr=0.5
        [158]	train_score: 2.306684	valid_score: 2.312795
        (100,80) lr=0.5
        [362]	train_score: 2.320007	valid_score: 2.329979
        (100,120) lr=0.5
        [348]	train_score: 2.312351	valid_score: 2.321759
        (100,150) lr=0.5
        [391]	train_score: 2.284677	valid_score: 2.295818

        first layer 800
        second layer 500, 800, 1500, 2000, 3000
        lr=0.5
        差别不大 均在2.275附近，曲线也近似

### xepa:
        ensemble_3_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 0.5]
            learning_rate = 0.2
            345	2.184544	2.184539	2.257029
        ensemble_4_mlp_1

## 2016/8/17, 19 days left

### rocky:
        bagofapps:
            mlp: valid logloss=2.259028
                 score = 2.24359


        concat_7_norm:
            gblinear:
            gbtree:
                max_depth=7, eta=0.05, subsample=0.8, colsample_bytree=0.5
                train_logloss=1.89917527015, valid_losloss=2.27997448448
                numofround=750
                score=2.26694
            mlp:
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 0.5]
                learning_rate = 0.2
                [444]	train_score: 2.211561	valid_score: 2.259309
                score=2.24387

        concat_6
            random_forest:
                n_estimators=1000 max_depth=40 max_features=0.1
                1.78502652928 2.30420502512
                score=2.29714

        blending:

### xepa:
        concat_6:
            fm:

        concat_7_norm:
            fm:

        concat_10_norm:
            fm:

### gym:
        ensemble_3_mlp_1:
            layer_sizes = [ti.SPACE, 90, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 0.5]
            learning_rate = 0.2
            num_round = 345
            327	2.189949	2.189951	2.257159
            score:2.24554

        concat_10_gblinear_1:
            alpha 0.1 lambda 45 2.16418639228 2.27902957404

        concat_10_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 0.5]
            learning_rate = 0.2
            num_round = 500
            461	2.206019	2.206028	2.258760

        concat_6_mpl_5:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 500
            [498]	loss: 2.197575 	train_score: 2.197590	valid_score: 2.258964	time: 5

        concat_11_freq_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 500
            [483]	train_score: 2.203397	valid_score: 2.258890

        concat_11, gbtree, {max_depth=3, eta=0.1, subsample=0.8, colsample=0.5, early_stop_round=20,
                                round=2000, lambda=1, alpha=0.1, rate=0.2}
                             valid_score=2.27878280383, real_score=not yet, concat_11_gbtree

        concat_12, gbtree, {max_depth=3, eta=0.1, subsample=0.7, colsample=0.7, early_stop_round=20,
                                round=2000, lambda=1, alpha=0.1, rate=0.2}
                             valid_score=2.27939683794, real_score=not yet, concat_12_gbtree

## 2016/8/18 Thu

### gym:
        concat_12_wrong_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 462
            best iteration:
            [462]	train_score: 2.199512	valid_score: 2.259978
            score:2.24657

        concat_11_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 600
            best iteration:
            [477]	train_score: 2.202244	valid_score: 2.258862
            score:2.24497

        concat_12_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 464
            best iteration:
            [464]	train_score: 2.201910	valid_score: 2.259253
            score:2.24769

        concat_13_gbtree_1:
            {max_depth=3, eta=0.1, subsample=0.6, colsample=0.9, early_stop_round=20,
                round=2000, lambda=1, alpha=0.1, rate=0.2}
                valid_score=2.28013701981, real_score=not yet, concat_13_gbtree

        concat_13_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 600
            best iteration:
            [398]	train_score: 2.203885	valid_score: 2.262226

        concat_14_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 600
            best iteration:
            [450]	train_score: 2.198796	valid_score: 2.260234

        concat_15_mlp_1:
            layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
            layer_activates = ['relu', None]
            drops = [0.5, 1]
            learning_rate = 0.2
            num_round = 600
            best iteration:
            [500]	train_score: 2.199825	valid_score: 2.258154
            score:2.24435

## 2016/8/18 Fri

### gym
            concat_15_mlp_2:
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 530
                score:2.24534

            concat_15_mlp_3:
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 470
                score:2.24394

            concat_16_mlp_1:
                cluster = 40
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 600
                best iteration:
                [505]	train_score:2.197933    valid_score:2.258077
                true_round = 480
                score:2.24449

            concat_16_2_mlp_1:
                cluster = 100
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 600
                best iteration:
                [496]	train_score: 2.197773	valid_score: 2.258149

            concat_16_3_mlp_1:
                cluster = 270
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 600
                best iteration:
                [476]	train_score: 2.197187	valid_score: 2.260958

            concat_17_mlp_1:
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 600
                best iteration:
                [471]	train_score: 2.202748	valid_score: 2.258381

            concat_17_mlp_2:
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.1
                num_round = 700
                best iteration:
                699	    2.231004	2.262013

            concat_16_tfidf_mlp_1:
                layer_sizes = [ti.SPACE, 100, ti.NUM_CLASS]
                layer_activates = ['relu', None]
                drops = [0.5, 1]
                learning_rate = 0.2
                num_round = 700
                best iteration:
                [489]	train_score: 2.201669	valid_score: 2.259127


## 2016/8/22 Mon

### gym
            concat_22_128_mlp_1:
                layer_sizes = [task.space, 64, task.num_class]
                learning_rate = 0.2
                batch_size = 1000
                num_round = 500
                
            concat_22_128_mlp_2:
                layer_sizes = [task.space, 64, 64, task.num_class]
                layer_activates = ['relu', 'relu', None]
                layer_inits = [('res:w0', 'res:b0'), ('res:pass', 'zero'), ('res:w1', 'res:b1')]
                init_path = '../model/concat_22_128_mlp_1.bin'
                layer_drops = [0.5, 0.75, 1]
                opt_algo = 'gd'
                learning_rate = 0.1
                num_round = 1000
                early_stop_round = 30
                
            concat_22_128_mlp_5:
                layer_sizes = [task.space, 64, task.num_class]
                learning_rate = 0.1
                batch_size = 128 256 512 1024 
                num_round = 1000
                
            concat_22_128_mlp_6:
                layer_sizes = [task.space, 64, task.num_class]
                learning_rate = 0.2
                batch_size = [32*2**p for p in range(7)]
            
            concat_15_mlp_1:
                learning_rate = 0.2
                num_round = 1000
                batch_size = [32*2**p for p in range(7)]
            
            concat_15_mlp_2:
                learning_rate = 0.1
                num_round = 1000
                batch_size = [32*2**p for p in range(10)]
            

## 2016/8/23 Tue

### gym
            concat_6_mnn_1:
                layer_sizes = [task.sub_spaces, [64, 64, 128, 64], task.num_class]
                layer_activates = ['relu', None]
                layer_inits = [('normal', 'zero'), ('normal', 'zero')]
                init_path = None
                layer_drops = [0.5, 1]
                opt_algo = 'gd'
                learning_rate = 0.1
                num_round = 1000
                early_stop_round = 20
                for batch_size in [1024, 2048, 4096, 8192, 64, 128, 256, 512]
                
            concat_6_mnn_2:
                layer_sizes = [task.sub_spaces, [64, 64, 128, 64], task.num_class]
                layer_activates = ['relu', None]
                layer_inits = [('normal', 'zero'), ('normal', 'zero')]
                init_path = None
                layer_drops = [0.5, 1]
                opt_algo = 'adam'
                num_round = 1000
                early_stop_round = 20
                batch_size = 1024
                for learning_rate in [0.1, 0.01, 0.001, 0.0001]
             
            
            版本3,4基于1 
            concat_7_norm_mlp_1:
                layer_sizes = [task.space, 100, task.num_class]
                opt_algo = 'gd'
                learning_rate = 0.2
                num_round = 1000
                early_stop_round = 30
                batch_size = 10000
                [90]	train_score: 2.206810	valid_score: 2.259412
            
            concat_7_norm_mlp_2:
                layer_sizes = [task.space, 100, task.num_class]
                opt_algo = 'gd'
                learning_rate = 0.2
                num_round = 1000
                early_stop_round = 10
                batch_size = 1000
                [49]	train_score: 2.208927	valid_score: 2.259177
            
            concat_7_norm_mlp_2继续训练的结果没有提高
            
            concat_7_norm_mlp_3:
                learning_rate = 0.0001
                layer_drops = [0.5, 0.75, 1]
                layer_sizes = [task.space, 128, 256, task.num_class]
                [3]	train_score: 2.203337	valid_score: 2.257690
          
            concat_7_norm_mlp_5:
                layer_sizes = [task.space, 128, task.num_class]
                layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
                init_path = '../model/concat_7_norm_mlp_4.bin'
                layer_drops = [0.5, 1]
                opt_algo = 'adam'
                learning_rate = 0.00001
                num_round = 1000
                
## 16/8/23

### xepa
        
        concat_6_mlp_107.submission
            layer_sizes = [task.space, 100, task.num_class]
            layer_activates = ['relu', None]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_6_mlp_100.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            batch_size = -1
            num_round = 180
            score = 2.24192
            
        concat_6_mlp_108.submission
            layer_sizes = [task.space, 100, task.num_class]
            layer_activates = ['relu', None]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_6_mlp_100.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            batch_size = -1
            num_round = 200
            score = 2.24151
            
        concat_6_mlp_109.submission
            layer_sizes = [task.space, 100, task.num_class]
            layer_activates = ['relu', None]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_6_mlp_100.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            batch_size = -1
            num_round = 220
            score = 2.24117
            
        concat_6_mlp_110.submission
            layer_sizes = [task.space, 100, task.num_class]
            layer_activates = ['relu', None]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_6_mlp_100.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            batch_size = -1
            num_round = 250
            score = 2.24100
           
        concat_6_mlp_110.submission
            layer_sizes = [task.space, 100, task.num_class]
            layer_activates = ['relu', None]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_6_mlp_100.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            batch_size = -1
            num_round = 300
            score = 2.24122
                early_stop_round = 5
                batch_size = 20000
                [66]	train_score: 2.166202	valid_score: 2.257378

     
### gym
        concat_15_mlp_2:
            layer_sizes = [task.space, 128, task.num_class]
            learning_rate = 0.2
            batch_size = 1000
            early_stop_round = 5
            [49]	train_score: 2.200459	valid_score: 2.259897
            
        concat_15_mlp_3:
            layer_sizes = [task.space, 128, 256, task.num_class]
            init_path = '../model/concat_15_mlp_2.bin'
            layer_drops = [0.5, 0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            early_stop_round = 3
            batch_size = 1000
            [2]	train_score: 2.226319	valid_score: 2.257795
                
        concat_15_mlp_4:
            layer_sizes = [task.space, 128, task.num_class]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_15_mlp_2.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.0001
            early_stop_round = 3
            batch_size = 1000
            [5]	train_score: 2.178618	valid_score: 2.256920
                
        concat_15_mlp_5:
            layer_sizes = [task.space, 128, task.num_class]
            layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1')]
            init_path = '../model/concat_15_mlp_4.bin'
            layer_drops = [0.5, 1]
            opt_algo = 'adam'
            learning_rate = 0.00001
            early_stop_round = 3
            batch_size = 20000
            [7]	train_score: 2.166192	valid_score: 2.256692
                
            
        concat_16_mlp_2:
            layer_sizes = [task.space, 128, task.num_class]
            learning_rate = 0.2
            batch_size = 10000
            early_stop_round = 10
            [445]	train_score: 2.204272	valid_score: 2.259139
                
        concat_16_mlp_3:
            layer_sizes = [task.space, 128, task.num_class]
            init_path = '../model/concat_16_mlp_2.bin'
            opt_algo = 'adam'
            learning_rate = 0.0001
            batch_size = 10000
            early_stop_round = 10
            [34]	train_score: 2.154118	valid_score: 2.256728
                
        concat_16_mlp_5:
            layer_sizes = [task.space, 256, task.num_class]
            init_path = '../model/concat_16_mlp_4.bin'
            opt_algo = 'adam'
            learning_rate = 0.00001
            batch_size = 10000
            early_stop_round = 10
            [4]	train_score: 2.154436	valid_score: 2.256839

            
        concat_15_mlp_1:
            layer_sizes = [task.space, 100, task.num_class]
            opt_algo = 'gd'
            learning_rate = 0.2
            batch_size = 10000
            [450]	train_score: 2.210901	valid_score: 2.258963
            [-1]	train_score: 2.208840	valid_score: 2.258979

                
## 2016/8/24 Wen

### gym
        concat_15_mlp_5:
            layer_sizes = [task.space, 100, task.num_class]
            init_path = '../model/concat_15_mlp_2.bin'
            opt_algo = 'adam'
            learning_rate = 0.0001
            early_stop_round = 5
            best iteration:
            [8]	loss: 2.161406 	train_score: 2.161406	valid_score: 2.255653
            [-1]	train_score: 2.161382	valid_score: 2.255770
                
        concat_15_mlp_6:
            new task.py
            same as concat_15_mlp_5
                 
        concat_16_mlp_3:
            layer_sizes = [task.space, 100, task.num_class]
            init_path = '../model/concat_16_mlp_2.bin'
            opt_algo = 'adam'
            learning_rate = 0.0001
            early_stop_round = 3
            best iteration:
            [31]	loss: 2.166870 	train_score: 2.166886	valid_score: 2.256226
            [-1]	train_score: 2.165275	valid_score: 2.256271
                
### xepa
        concat_6_mlp_138
            400
            score = 2.24015
            
        concat_6_mlp_139
            420
            score = 2.24001
            
        concat_6_mlp_140
            450
            score = 2.24002
            
        concat_6_mlp_141
            470
            score = 2.24003
            
---
            
### rocky

**embedding**

    phone_brand:
        (0.2, 64)       {[2121]	train_score: 2.402465	valid_score: 2.402623}      phone_brand_multi_layer_perceptron_1.bin
        (0.2, 128)      {[1825]	train_score: 2.401376	valid_score: 2.402689}      phone_brand_mlp_3.bin
        (0.2, 32)       {[1846]	train_score: 2.403960	valid_score: 2.403217}      phone_brand_multi_layer_perceptron_2
    device_model:
        (0.2, 64)       {[2652]	train_score: 2.369331	valid_score: 2.389789}      device_model_multi_layer_perceptron_1.bin
        (0.2, 32)       {[2891]	train_score: 2.374589	valid_score: 2.389791}
        (0.2, 128)      {[2288]	train_score: 2.368377	valid_score: 2.390303}

    phone_brand:
        gbtree      [113]	train-mlogloss:2.396648	eval-mlogloss:2.402918
    phone_brand_embedding_32
        gbtree      [51]	train-mlogloss:2.394280	eval-mlogloss:2.402758
    phone_brand_embedding_64
        gbtree      [49]	train-mlogloss:2.393891	eval-mlogloss:2.4027
    phone_brand_embedding_128
        gbtree      [52]	train-mlogloss:2.393246	eval-mlogloss:2.403260

    installed_app:
        mlp (0.2, 64)       [657]	train_score: 2.247402	valid_score: 2.287982      installed_app_mlp_1.bin
        embedding from this model   installed_app_embedding_1
    installed_app_label
        mlp (0.2, 64)       [687]	train_score: 2.308083	valid_score: 2.320131       installed_app_label_mlp_1.bin
        embedding from this model   installed_app_label_embedding_1

**concat_6_embedding_64**

    mlp (0.2, 64)       [296]	train_score: 2.210399	valid_score: 2.264480       concat_6_embedding_64_mlp_1.bin
                                submission score=2.25093
    mlp bypass (0.1, 64, 64)        [287]	train_score: 2.202804	valid_score: 2.261038
    mlp bypass (0.1, 64, 128)       [292]	train_score: 2.202365	valid_score: 2.260960   concat_6_embedding_64_mlp_3.bin
    mlp bypass (0.1, 128, 64)       [271]	train_score: 2.203536	valid_score: 2.261078
    mlp bypass (0.1, 128, 128)      [320]	train_score: 2.202656	valid_score: 2.261202
    based on concat_6_embedding_64_mlp_3.bin
    layer_sizes = [task.space, 64, 64, 128,  task.num_class]
    layer_inits = [('res:w0', 'res:b0'), ('res:pass', 'zero'), ('res:w1', 'res:b1'), ('res:w2', 'res:b2')]
    [0]	train_score: 2.213178	valid_score: 2.261509
    layer_sizes = [task.space, 64, 128, 128,  task.num_class]
    layer_inits = [('res:w0', 'res:b0'), ('res:pass', 'zero'), ('res:w1', 'res:b1'), ('res:w2', 'res:b2')]
    [0]	train_score: 2.215293	valid_score: 2.261441
    layer_sizes = [task.space, 64, 128, 128,  task.num_class]
    layer_inits = [('res:w0', 'res:b0'), ('res:w1', 'res:b1'), ('res:pass', 'zero'), ('res:w2', 'res:b2')]
    [0]	train_score: 2.213601	valid_score: 2.261232


    mlp (0.2, 128)      [232]	train_score: 2.206101	valid_score: 2.264820       concat_6_embedding_64_mlp_5.bin
    mlp bypass (0.1, 128, 128)      [251]	train_score: 2.195242	valid_score: 2.262748

    mlp (0.2, 32)       [329]	train_score: 2.225020	valid_score: 2.265452

**concat_6_ooee_64**

    mlp (0.2, 64)       [392]	train_score: 2.212981	valid_score: 2.267150

**cocnat_21**

    mlp{gd}
    1    (0.2, 64), batch_size=-1   , drops=(0.5, 1)     {[3076]train_score: 2.207119	valid_score: 2.258043}
    2    (0.2, 64), batch_size=10000, drops=(0.5, 1)     {[557]	train_score: 2.201392	valid_score: 2.257924}
    3    (0.2, 64), batch_size=10000, drops=(0.75,1)     {[441]	train_score: 2.198582	valid_score: 2.260310}
    4    (0.2, 64), batch_size=10000, drops=(0.25,1)     {[659]	train_score: 2.226501	valid_score: 2.260723}
    5    (0.2, 64), batch_size=1000 , drops=(0.5, 1)     {[56]	train_score: 2.207921	valid_score: 2.260046}
    6    (0.2, 64), batch_size=5000 , drops=(0.5, 1)     {[265]	train_score: 2.207076	valid_score: 2.258683}
    7    (0.2, 64), batch_size=15000, drops=(0.5, 1)     {[786]	train_score: 2.204690	valid_score: 2.259189}
    8    (0.2, 64), batch_size=20000, drops=(0.5, 1)     {[1074]train_score: 2.204936	valid_score: 2.258583}
    9    (0.1, 64), batch_size=1000 , drops=(0.5, 1)     {[118]	train_score: 2.195672	valid_score: 2.258665}
    10   (0.05,64), batch_size=1000 , drops=(0.5, 1)     {[203]	train_score: 2.206909	valid_score: 2.258858}
    11   (0.1, 64), batch_size=10000, drops=(0.5, 1)     {[1077]train_score: 2.200618	valid_score: 2.257873}      concat_21_mlp_1.bin
    12   (0.1,128), batch_size=10000, drops=(0.5, 1)     {[939]	train_score: 2.199193	valid_score: 2.258999}
    mlp{adam}
        (1e-4,64), batch_size=-1   , drops=(0.5, 1)     {[71]	train_score: 2.173218	valid_score: 2.256882}      submission score: 2.24364

    一层：
    曲线大概已经对应不上了。单从结果来看。
    batch_size：bs越大，步长越小，需要训练的轮数越多，对收敛结果的影响不规则，不同bs造成的影响差距感觉更来源于参数初始化时候的不确定性，
                理由为在参数（0.2， 64， 10000）条件下，多次训练有时也无法达到2.2579+ ， 对比1,2,5,6,7,8. 2中结果最好，1次之
    drop out  :drops取个中间值感觉较好，对比2,3,4分别取值为0.5,0.25,0.75 其中取值0.5时效果较好，不同取值差距较大。
    learning rate：lr越小，需要训练的轮数越多。对比5,9,10大概是lr取小一点会稍微好一点
    神经元个数 ：对比11,12。64一直以来比128（以及32）好点
    在用gd训练好后，用它的参数再取比较小lr用adam算法可以有一点点提升。
    二层：
    大概是加不上去了

**ensemble_6**

    mlp
        (0.2, 64), batch_size=10000, drops=(0.5, 1)     {[154]	train_score: 2.117223	valid_score: 2.265209}      ensemble_6_mlp_1.bin
        (0.2, 64), batch_size=10000, drops=(0.5, 1)     {[150]	train_score: 2.120430	valid_score: 2.264654}
        (0.2, 64), batch_size=-1   , drops=(0.5, 1)     {电脑瘫痪，大概数据稠密，内存爆了？？}
        (0.1, 64), batch_size=4096 , drops=(0.5, 1)     {[302]	train_score: 2.122921	valid_score: 2.266524}
        

## 2016/8/25 Thu

### xepa
    300 - 349: 5 fold, 10 repeats
    350 - 359: full set, 10 repeats
    concat_6_mlp_360.submission
    2.24381
    
    concat_6_mlp_361.submission
    2.24399
    
    concat_6_mlp_362.submission
    2.24312
    
    
## 2016/8/26 Fri

### gym:
    
    ensemble_6_mnn
        layer_l2 = [0.1, 0.3]
        batch_sizes = 1000
        learning_rate = 0.0001
        
    concat_7_norm_mlp:
        layer_l2 = [0, 0.01]
        layer_sizes = [task.space, 72, task.num_class]
        init_path = '../model/concat_7_norm_mlp_2.bin'
        layer_drops = [0.6, 1]
        layer_l2 = [0, 0.01]
        opt_algo = 'adam'
        learning_rate = 0.0001
        batch_size = 10000
        [95]	loss: 2.262365 	train_score: 2.150089	valid_score: 2.254761
        
        submission_1:
            num_round = 110
            [109]	loss: 2.263455	train_score: 2.170605
         
        submission_2:
            num_round = 116
            [115]	loss: 2.258084	train_score: 2.167240
            socre: 2.24152
        
        submission_3:
            num_round = 126
            [115]	loss: 2.258084	train_score: 2.167240
        
        submission_4:
            num_round = 135
            [134]	loss: 2.244782	train_score: 2.159963
            score:2.24151
            
        submission_5:
            num_round = 140
            [139]	loss: 2.240977	train_score: 2.157368
            score: 2.24166
            
       
## 2016/8/27 Sat

### gym:
    
     ensemble_7_mlp:
        random_seed = 0x4567
        layer_sizes = [task.space, 100, task.num_class]
        layer_l2 = [0, 0.01]
        opt_algo = 'adam'
        learning_rate = 0.0001
        layer_drops = [0.5, 1]
        batch_size = 1000
        [25]	loss: 2.090874 	train_score: 2.071044	valid_score: 2.254539
        
        submission_1:
            num_round = 25
            [24]	loss: 2.116322	train_score: 2.096564
            score:2.24347
            
        submission_2:
            num_round = 28
            score:2.24513
            
            
## 16/8/28

### xepa
    
    params = {
        'layer_sizes': [task.space, 100, task.num_class],
        'layer_activates': ['relu', None],
        'layer_drops': [0.5, 1],
        'layer_l2': [0.0001, 0.0001],
        'layer_inits': [('res:w0', 'res:b0'), ('res:w1', 'res:b1')],
        'init_path': '../model/concat_6_mlp_100.bin',
        'opt_algo': 'adam',
        'learning_rate': 0.0001,
        'random_seed': 0x0123,
    }
    batch_size = -1
    num_round = 500
    early_stop_round = 10
    
    0x0123
    [215]	train_score: 2.161063	valid_score: 2.254976
    [-1]	train_score: 2.157810	valid_score: 2.254982
    model dumped at ../model/concat_6_mlp_141.bin
    
    0x4567
    [214]	train_score: 2.161927	valid_score: 2.255016
    [-1]	train_score: 2.158975	valid_score: 2.255033
    model dumped at ../model/concat_6_mlp_142.bin
    
    0x89AB
    [233]	train_score: 2.155918	valid_score: 2.254898
    [-1]	train_score: 2.153363	valid_score: 2.254970
    model dumped at ../model/concat_6_mlp_143.bin
    
    0xCDEF
    [225]	train_score: 2.157516	valid_score: 2.254989
    [-1]	train_score: 2.155495	valid_score: 2.255020
    model dumped at ../model/concat_6_mlp_144.bin
    
    0x3210
    [223]	train_score: 2.157971	valid_score: 2.254946
    [-1]	train_score: 2.155180	valid_score: 2.254974
    model dumped at ../model/concat_6_mlp_145.bin
    
    0x7654
    [208]	train_score: 2.162757	valid_score: 2.254969
    [-1]	train_score: 2.160468	valid_score: 2.255000
    model dumped at ../model/concat_6_mlp_146.bin
    
    0xBA98
    [202]	train_score: 2.164088	valid_score: 2.255046
    [-1]	train_score: 2.163134	valid_score: 2.255051
    model dumped at ../model/concat_6_mlp_147.bin
    
    0xFEDC
    [227]	train_score: 2.157217	valid_score: 2.254903
    [-1]	train_score: 2.154929	valid_score: 2.254913
    model dumped at ../model/concat_6_mlp_148.bin

### gym

    concat_7_norm_mlp_2:
        layer_sizes = [task.space, 100, task.num_class]
        init_path = '../model/concat_6_mlp_143.bin'
        layer_drops = [0.2, 1]
        layer_l2 = [0.0001, 0.0001]
        opt_algo = 'adam'
        learning_rate = 1e-6
        random_seed = 0x89AB
        best iteration:
        [681]	loss: 2.212628 	train_score: 2.198552	valid_score: 2.254312
    
        layer_drops = [0.4, 1]
        best iteration:
        [278]	loss: 2.174715 	train_score: 2.160659	valid_score: 2.254841
        
        
    concat_7_norm_mlp_3:
        layer_sizes = [task.space, 128, task.num_class]
        layer_inits = [('net2:w0', 'net2:b0'), ('net2:w1', 'net2:b1')]
        init_path = '../model/concat_6_mlp_143.bin'
        layer_drops = [0.5, 1]
        layer_l2 = [0.0001, 0.0001]
        opt_algo = 'adam'
        learning_rate = 1e-5
        random_seed = 0x89AB
        batch_size = -1
        early_stop_round = 20
        best iteration:
        [474]	loss: 2.164691 	train_score: 2.150694	valid_score: 2.254168
        
    concat_7_norm_mlp_31.submission
        num_round = 500
        [499]	loss: 2.184437	train_score: 2.170846
    
    concat_7_norm_mlp_32.submission
        num_round = 600
        [599]	loss: 2.179643	train_score: 2.165545
        
    concat_7_norm_mlp_33.submission
        num_round = 900
        [899]	loss: 2.172942	train_score: 2.157660
    
    concat_7_norm_mlp_34.submission
        num_round = 1000
        [999]	loss: 2.171967	train_score: 2.156395	
        
    concat_7_norm_mlp_35.submission
        num_round = 1500
        [1499]	loss: 2.158979	train_score: 2.140908
        
    concat_7_norm_mlp_36.submission
        num_round = 1111
        [1110]	loss: 2.167221	train_score: 2.150921
        
    concat_7_norm_mlp_37.submission
        num_round = 1125
        [1124]	loss: 2.167855	train_score: 2.151651