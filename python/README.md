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


### ensemble features

    emsemble_1, multi, 42215, 5890
    
    ensemble_2
    
    ensemble_3: [concat_1_gbtree_1, concat_1_gblinear_1, concat_6]
    
    ensemble_4, [fea_concat_1_gbtree_1,fea_concat_1_gblinear_1,fea_concat_2_gbtree_1,fea_concat_2_gblinear_1,fea_concat_2_norm_gbtree_1,fea_concat_2_norm_gblinear_1,fea_concat_3_gbtree_1,fea_concat_3_gblinear_1,fea_concat_3_norm_gbtree_1,fea_concat_3_norm_gblinear_1,fea_concat_4_gbtree_1,fea_concat_4_gblinear_1,fea_concat_4_norm_gbtree_1,fea_concat_4_norm_gblinear_1,fea_concat_5_gbtree_1,fea_concat_5_gblinear_1,# fea_concat_5_norm_gbtree_1,# fea_concat_5_norm_gblinear_1,fea_concat_6_gbtree_1,fea_concat_6])
        
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
