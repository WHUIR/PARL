import math


def get_test_set2(testRatings_list, user_reviews, item_reviews, user_aux_reviews, user_masks, item_masks, user_aux_masks):
    user_test, item_test, user_input_test, item_input_test, user_aux_input_test, user_mask_input_test, item_mask_input_test, user_aux_mask_input_test, rating_input_test = [], [], [], [], [], [], [], [], []
    for idx in xrange(len(testRatings_list)):
        ratings = testRatings_list[idx]
        user = ratings[0]
        item = ratings[1]
        rate = ratings[2]
        user_test.append(user)
        item_test.append(item)
        user_input_test.append(user_reviews.get(user))
        item_input_test.append(item_reviews.get(item))
        user_aux_input_test.append(user_aux_reviews.get(user))
        user_mask_input_test.append(user_masks.get(user))
        item_mask_input_test.append(item_masks.get(item))
        user_aux_mask_input_test.append(user_aux_masks.get(user))
        rating_input_test.append(rate)
    return user_test, item_test, user_input_test, item_input_test, user_aux_input_test, user_mask_input_test, item_mask_input_test, user_aux_mask_input_test, rating_input_test


def get_train_instance_batch_test_aux(count, batch_size, user, item, user_input, item_input, user_aux_input, user_mask, item_mask, user_aux_mask, ratings):
    user_batch, item_batch, user_input_batch, item_input_batch, user_aux_input_batch, user_mask_batch, item_mask_batch, user_aux_mask_batch, labels_batch = [], [], [], [], [], [], [], [], []

    for idx in xrange(batch_size):
        index = count*batch_size + idx
        if (index >= len(user_input)):
            break
        user_batch.append(user[index])
        item_batch.append(item[index])
        user_input_batch.append(user_input[index])
        user_mask_batch.append(user_mask[index])
        item_input_batch.append(item_input[index])
        item_mask_batch.append(item_mask[index])
        user_aux_input_batch.append(user_aux_input[index])
        user_aux_mask_batch.append(user_aux_mask[index])
        labels_batch.append(ratings[index])

    return user_batch, item_batch, user_input_batch, item_input_batch, user_aux_input_batch, user_mask_batch, item_mask_batch, user_aux_mask_batch, labels_batch
