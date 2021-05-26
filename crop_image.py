import sys
import cv2
import numpy as np
from collections import Counter
from copy import deepcopy

def get_ref_lengths(img):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    rle_image_white_runs = []  # Cumulative white run list
    rle_image_black_runs = []  # Cumulative black run list
    sum_all_consec_runs = []  # Cumulative consecutive black white runs

    for i in range(num_cols):
        col = img[:, i]
        rle_col = []
        rle_white_runs = []
        rle_black_runs = []
        run_val = 0  # (The number of consecutive pixels of same value)
        run_type = col[0]  # Should be 255 (white) initially
        for j in range(num_rows):
            if (col[j] == run_type):
                # increment run length
                run_val += 1
            else:
                # add previous run length to rle encoding
                rle_col.append(run_val)
                if (run_type == 0):
                    rle_black_runs.append(run_val)
                else:
                    rle_white_runs.append(run_val)

                # alternate run type
                run_type = col[j]
                # increment run_val for new value
                run_val = 1

        # add final run length to encoding
        rle_col.append(run_val)
        if (run_type == 0):
            rle_black_runs.append(run_val)
        else:
            rle_white_runs.append(run_val)

        # Calculate sum of consecutive vertical runs
        sum_rle_col = [sum(rle_col[i: i + 2]) for i in range(len(rle_col))]

        # Add to column accumulation list
        rle_image_white_runs.extend(rle_white_runs)
        rle_image_black_runs.extend(rle_black_runs)
        sum_all_consec_runs.extend(sum_rle_col)

    white_runs = Counter(rle_image_white_runs)
    black_runs = Counter(rle_image_black_runs)
    black_white_sum = Counter(sum_all_consec_runs)

    line_spacing = white_runs.most_common(1)[0][0]
    line_width = black_runs.most_common(1)[0][0]
    width_spacing_sum = black_white_sum.most_common(1)[0][0]

    assert (line_spacing + line_width == width_spacing_sum), "Estimated Line Thickness + Spacing doesn't correspond with Most Common Sum "

    return line_width, line_spacing

#######################################################


def find_staffline_rows(img, line_width, line_spacing):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    row_black_pixel_histogram = []

    # Determine number of black pixels in each row
    for i in range(num_rows):
        row = img[i]
        num_black_pixels = 0
        for j in range(len(row)):
            if (row[j] == 0):
                num_black_pixels += 1

        row_black_pixel_histogram.append(num_black_pixels)

    # plt.bar(np.arange(num_rows), row_black_pixel_histogram)
    # plt.show()

    all_staff_row_indices = []
    num_stafflines = 5
    threshold = 0.4
    staff_length = num_stafflines * (line_width + line_spacing) - line_spacing
    iter_range = num_rows - staff_length + 1

    # Find stafflines by finding sum of rows that occur according to
    # staffline width and staffline space which contain as many black pixels
    # as a thresholded value (based of width of page)
    #
    # Filter out using condition that all lines in staff
    # should be above a threshold of black pixels
    current_row = 0
    while (current_row < iter_range):
        staff_lines = [row_black_pixel_histogram[j: j + line_width] for j in
                       range(current_row, current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
                             line_width + line_spacing)]
        pixel_avg = sum(sum(staff_lines, [])) / (num_stafflines * line_width)

        for line in staff_lines:
            if (sum(line) / line_width < threshold * num_cols):
                current_row += 1
                break
        else:
            staff_row_indices = [list(range(j, j + line_width)) for j in
                                 range(current_row,
                                       current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
                                       line_width + line_spacing)]
            all_staff_row_indices.append(staff_row_indices)
            current_row = current_row + staff_length

    return all_staff_row_indices


def find_staffline_columns(img, all_staffline_vertical_indices, line_width, line_spacing):
    num_rows = img.shape[0]  # Image Height (number of rows)
    num_cols = img.shape[1]  # Image Width (number of columns)
    # Create list of tuples of the form (column index, number of occurrences of width_spacing_sum)
    all_staff_extremes = []

    # Find start of staff for every staff in piece
    for i in range(len(all_staffline_vertical_indices)):
        begin_list = [] # Stores possible beginning column indices for staff
        end_list = []   # Stores possible end column indices for staff
        begin = 0
        end = num_cols - 1

        # Find staff beginning
        for j in range(num_cols // 2):
            first_staff_rows_isolated = img[all_staffline_vertical_indices[i][0][0]:all_staffline_vertical_indices[i][4][
                line_width - 1], j]
            num_black_pixels = len(list(filter(lambda x: x == 0, first_staff_rows_isolated)))

            if (num_black_pixels == 0):
                begin_list.append(j)

        # Find maximum column that has no black pixels in staff window
        list.sort(begin_list, reverse=True)
        begin = begin_list[0]

        # Find staff beginning
        for j in range(num_cols // 2, num_cols):
            first_staff_rows_isolated = img[all_staffline_vertical_indices[i][0][0]:all_staffline_vertical_indices[i][4][
                line_width - 1], j]
            num_black_pixels = len(list(filter(lambda x: x == 0, first_staff_rows_isolated)))

            if (num_black_pixels == 0):
                end_list.append(j)

        # Find maximum column that has no black pixels in staff window
        list.sort(end_list)
        end = end_list[0]

        staff_extremes = (begin, end)
        all_staff_extremes.append(staff_extremes)

    return all_staff_extremes


###############################################################################


def default(arg_image): 
    img = cv2.imread(arg_image,0) 
    retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    line_width, line_spacing = get_ref_lengths(img)

    print("[INFO] Staff line Width: ", line_width)
    print("[INFO] Staff line Spacing: ", line_spacing)

    # ============ Find Staff Line Rows ============

    all_staffline_vertical_indices = find_staffline_rows(img, line_width, line_spacing)
    print("[INFO] Found ", len(all_staffline_vertical_indices), " sets of staff lines")

    # ============ Find Staff Line Columns ============

    # Find column with largest index that has no black pixels

    all_staffline_horizontal_indices = find_staffline_columns(img, all_staffline_vertical_indices, line_width, line_spacing)
    print("[INFO] Found all staff line horizontal extremes")

    #######################################################################

    # ============ Show Detected Staffs ============
    staffs = []
    half_dist_between_staffs = (all_staffline_vertical_indices[1][0][0] - all_staffline_vertical_indices[0][4][line_width - 1])//2

    for i in range(len(all_staffline_vertical_indices)):
        x = all_staffline_horizontal_indices[i][0]
        y = all_staffline_vertical_indices[i][0][0]
        width = all_staffline_horizontal_indices[i][1] - x
        height = all_staffline_vertical_indices[i][4][line_width - 1] - y
        # Create Cropped Staff Image
        staff_img = img[max(0, y - half_dist_between_staffs + height): min(y + half_dist_between_staffs, img.shape[0] - 1), x:x+width]
        staffs.append(staff_img)
    return staffs



###############################################################################

# import sys
# import cv2
# import numpy as np
# from collections import Counter
# from copy import deepcopy
# import omr_utils
# import matplotlib.pyplot as plt

# def get_ref_lengths(img):
#     num_rows = img.shape[0]  # Image Height (number of rows)
#     num_cols = img.shape[1]  # Image Width (number of columns)
#     rle_image_white_runs = []  # Cumulative white run list
#     rle_image_black_runs = []  # Cumulative black run list
#     sum_all_consec_runs = []  # Cumulative consecutive black white runs

#     for i in range(num_cols):
#         col = img[:, i]
#         rle_col = []
#         rle_white_runs = []
#         rle_black_runs = []
#         run_val = 0  # (The number of consecutive pixels of same value)
#         run_type = col[0]  # Should be 255 (white) initially
#         for j in range(num_rows):
#             if (col[j] == run_type):
#                 # increment run length
#                 run_val += 1
#             else:
#                 # add previous run length to rle encoding
#                 rle_col.append(run_val)
#                 if (run_type == 0):
#                     rle_black_runs.append(run_val)
#                 else:
#                     rle_white_runs.append(run_val)

#                 # alternate run type
#                 run_type = col[j]
#                 # increment run_val for new value
#                 run_val = 1

#         # add final run length to encoding
#         rle_col.append(run_val)
#         if (run_type == 0):
#             rle_black_runs.append(run_val)
#         else:
#             rle_white_runs.append(run_val)

#         # Calculate sum of consecutive vertical runs
#         sum_rle_col = [sum(rle_col[i: i + 2]) for i in range(len(rle_col))]

#         # Add to column accumulation list
#         rle_image_white_runs.extend(rle_white_runs)
#         rle_image_black_runs.extend(rle_black_runs)
#         sum_all_consec_runs.extend(sum_rle_col)

#     white_runs = Counter(rle_image_white_runs)
#     black_runs = Counter(rle_image_black_runs)
#     black_white_sum = Counter(sum_all_consec_runs)

#     line_spacing = white_runs.most_common(1)[0][0]
#     line_width = black_runs.most_common(1)[0][0]
#     width_spacing_sum = black_white_sum.most_common(1)[0][0]

#     assert (line_spacing + line_width == width_spacing_sum), "Estimated Line Thickness + Spacing doesn't correspond with Most Common Sum "

#     return line_width, line_spacing

# #######################################################


# def find_staffline_rows(img, line_width, line_spacing):
#     num_rows = img.shape[0]  # Image Height (number of rows)
#     num_cols = img.shape[1]  # Image Width (number of columns)
#     row_black_pixel_histogram = []

#     # Determine number of black pixels in each row
#     for i in range(num_rows):
#         row = img[i]
#         num_black_pixels = 0
#         for j in range(len(row)):
#             if (row[j] == 0):
#                 num_black_pixels += 1

#         row_black_pixel_histogram.append(num_black_pixels)

#     # plt.bar(np.arange(num_rows), row_black_pixel_histogram)
#     # plt.show()
#     all_staff_row_indices = []
#     num_stafflines = 5
#     staff_length = (num_stafflines * (line_width + line_spacing)) - line_spacing
#     black_length = 0
#     print(len(row_black_pixel_histogram) ,  staff_length)
#     tor = []
#     for i in row_black_pixel_histogram:
#         if i > 0.7 * num_cols:
#             tor.append(i)
#     print(len(tor))
#     while  black_length < len(row_black_pixel_histogram):
#         if row_black_pixel_histogram[black_length] > 0.7 * num_cols :
#             print(row_black_pixel_histogram[black_length])
#             staff_row_indices = [[black_length]]
#             for i in range(4):
#                 staff_row_indices.append([black_length + (i + 1 ) * (line_width + line_spacing)] )
#             all_staff_row_indices.append(staff_row_indices)
#             black_length += staff_length
#         else:
#             black_length += 1
#     return all_staff_row_indices

#     all_staff_row_indices = []
 
#     threshold = 0.4
   
#     iter_range = num_rows - staff_length + 1
#     # print("iter-Range" , iter_range)

#     # Find stafflines by finding sum of rows that occur according to
#     # staffline width and staffline space which contain as many black pixels
#     # as a thresholded value (based of width of page)
#     #
#     # Filter out using condition that all lines in staff
#     # should be above a threshold of black pixels
#     current_row = 0
#     while (current_row < iter_range):
#         # print(current_row , iter_range)
#         staff_lines = [row_black_pixel_histogram[j: j + line_width] for j in
#                        range(current_row, current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
#                              line_width + line_spacing)]
#         staff_lines2 = [row_black_pixel_histogram[j: j + line_width] for j in
#                        range((current_row + 1), ( current_row + 1 ) + (num_stafflines - 1) * (line_width + line_spacing) + 1,
#                              line_width + line_spacing)]
#         # pixel_avg = sum(sum(staff_lines, [])) / (num_stafflines * line_width)


#         # new_lines = []
#         mean_lines = 0
#         # if current_row + 1 == iter_range:
#         #     new_lines = staff_lines
#         # else:
#         #     for lx in range(len(staff_lines)):
#         #         new_lines.append([max(staff_lines[lx][0] , staff_lines2[lx][0])])

#         new_lines = staff_lines

#         # print(new_lines , threshold * num_cols)
#         for lh in new_lines:
#             mean_lines += sum(lh)
#         # for line in new_lines:
#         #     if (sum(line) / line_width < threshold * num_cols):
#         #         print(line , threshold * num_cols, sum(line) / line_width < threshold * num_cols)
#         #         current_row += 1
#         #         break
#         if ( mean_lines // 4) < threshold * num_cols:
#             current_row += 4
#         else:
#             print('************************', current_row)
#             staff_row_indices = [list(range(j, j + line_width)) for j in
#                                  range(current_row,
#                                        current_row + (num_stafflines - 1) * (line_width + line_spacing) + 1,
#                                        line_width + line_spacing)]
#             all_staff_row_indices.append(staff_row_indices)
#             current_row = current_row + staff_length
#             print(current_row)

#     return all_staff_row_indices


# def find_staffline_columns(img, all_staffline_vertical_indices, line_width, line_spacing):
#     num_rows = img.shape[0]  # Image Height (number of rows)
#     num_cols = img.shape[1]  # Image Width (number of columns)
#     # Create list of tuples of the form (column index, number of occurrences of width_spacing_sum)
#     all_staff_extremes = []

#     # Find start of staff for every staff in piece
#     for i in range(len(all_staffline_vertical_indices)):
#         begin_list = [] # Stores possible beginning column indices for staff
#         end_list = []   # Stores possible end column indices for staff
#         begin = 0
#         end = num_cols - 1

#         # Find staff beginning
#         for j in range(num_cols // 2):
#             first_staff_rows_isolated = img[all_staffline_vertical_indices[i][0][0]:all_staffline_vertical_indices[i][4][
#                 line_width - 1], j]
#             num_black_pixels = len(list(filter(lambda x: x == 0, first_staff_rows_isolated)))

#             if (num_black_pixels == 0):
#                 begin_list.append(j)

#         # Find maximum column that has no black pixels in staff window
#         list.sort(begin_list, reverse=True)
#         begin = begin_list[0]

#         # Find staff beginning
#         for j in range(num_cols // 2, num_cols):
#             first_staff_rows_isolated = img[all_staffline_vertical_indices[i][0][0]:all_staffline_vertical_indices[i][4][
#                 line_width - 1], j]
#             num_black_pixels = len(list(filter(lambda x: x == 0, first_staff_rows_isolated)))

#             if (num_black_pixels == 0):
#                 end_list.append(j)

#         # Find maximum column that has no black pixels in staff window
#         list.sort(end_list)
#         end = end_list[0]

#         staff_extremes = (begin, end)
#         all_staff_extremes.append(staff_extremes)

#     return all_staff_extremes


# ###############################################################################


# def default(arg_image): 
#     img = cv2.imread(arg_image,0) 
#     retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     plt.imshow(img), plt.title('test')
#     plt.show()
#     line_width, line_spacing = get_ref_lengths(img)

#     print("[INFO] Staff line Width: ", line_width)
#     print("[INFO] Staff line Spacing: ", line_spacing)

#     # ============ Find Staff Line Rows ============

#     all_staffline_vertical_indices = find_staffline_rows(img, line_width, line_spacing)
#     print(all_staffline_vertical_indices)
#     print("[INFO] Found ", len(all_staffline_vertical_indices), " sets of staff lines")

#     # ============ Find Staff Line Columns ============

#     # Find column with largest index that has no black pixels

#     all_staffline_horizontal_indices = find_staffline_columns(img, all_staffline_vertical_indices, line_width, line_spacing)
#     print("[INFO] Found all staff line horizontal extremes")

#     #######################################################################

#     # ============ Show Detected Staffs ============
#     staffs = []
#     half_dist_between_staffs = (all_staffline_vertical_indices[1][0][0] - all_staffline_vertical_indices[0][4][line_width - 1])//2
#     for i in range(len(all_staffline_vertical_indices)):
#         x = all_staffline_horizontal_indices[i][0]
#         y = all_staffline_vertical_indices[i][0][0]
#         width = all_staffline_horizontal_indices[i][1] - x
#         height = all_staffline_vertical_indices[i][4][line_width - 1] - y
#         # Create Cropped Staff Image
#         staff_img = img[max(0, y - half_dist_between_staffs + (height // 2)): min(y + half_dist_between_staffs + (height //2 ), img.shape[0] - 1), x:x+width]
#         staffs.append(staff_img)
#         plt.imshow(staff_img, cmap='gray'), plt.title('staff image {}'.format(i+1))
#         plt.show()
#     return staffs