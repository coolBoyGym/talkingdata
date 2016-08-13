import train_impl as ti

ti.init_constant(dataset='concat_5', booster='gbtree', version=2, random_state=0)

if __name__ == '__main__':
    ti.train_gbtree_find_argument('../output/argument.concat_5.gbtree')

    # train_gblinear_find_argument('../output/argument.concat_3.gblinear')
    # train_gblinear_confirm_argument(0.2, 0.86, True)
    # train_gblinear_get_result(3, 0.2, 0.86)
    # print('main function finish!')
    ti.train_gbtree_find_argument('../output/new_arg_concat_5_gbtree.argument')
