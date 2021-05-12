library(ggplot2)
library(dplyr)

calcPrecision = function(tp, fp){
  precision = nrow(tp)/(nrow(tp)+nrow(fp))
  return(precision)
}

calcRecall = function(tp, fn){
  recall = nrow(tp)/(nrow(tp)+nrow(fn))
  return(recall)
}

binnedPR = function(pred_df, obs_df, target_col, bin_list){
  output_df = data.frame('bins' = bin_list, 'precision' = 0, 'recall' = 0)
  for (i in bin_list){
    current_df = subset(pred_df, target_col == i)
    tp = subset(current_df, gt_uid > 0 & y_pred == 1)
    tn = subset(current_df, gt_uid == 0 & y_pred == 0)
    fp = subset(current_df, gt_uid == 0 & y_pred == 1)
    fn = subset(obs_df, gt_uid > 0 & y_pred == 0)
    precision = calcPrecision(tp, fp)
    recall = calcRecall(tp, fn)
    output_df$precision[output_df$bins == i] = precision
    output_df$recall[output_df$bins == i] = recall
  }
  return(output_df)
}

unique_plots = unique(pred$plot)

test = binnedPR(pred, obs, pred$plot, unique_plots)

plot_df = data.frame('precision' = test$precision, 'recall' = test$recall, 'plot' = as.factor(test$bins))

ggplot(plot_df, aes(x=plot,y=precision)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_precision)
ggplot(plot_df, aes(x=plot,y=recall)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_recall)

pred_csv = '/Users/theo/data/hough_dataset/predicted_test_cleaned.csv'
ground_csv = '/Users/theo/data/hough_dataset/predicted_test.csv'

pred = read.csv(pred_csv)
obs = read.csv(ground_csv)

ground_truth = subset(obs, gt_uid > 0)
ground_false = subset(obs, gt_uid == 0)

obs$size = ntile(pred$r, 5)
obs$density = 0
pred$density = 0
plots = as.data.frame(table(ground_truth$plot))
for (i in plots$Var1){
  count = plots[plots$Var1 == i, 2]
  pred$density[pred$plot == i] = count
  obs$density[obs$plot == i] = count
}

tp = subset(pred, gt_uid > 0 & y_pred == 1)
tn = subset(pred, gt_uid == 0 & y_pred == 0)
fp = subset(pred, gt_uid == 0 & y_pred == 1)
fn = subset(obs, gt_uid > 0 & y_pred == 0)

avg_precision = calcPrecision(tp, fp)
p1 = calcPrecision(subset(tp, size == 1), subset(fp, size == 1))
p2 = calcPrecision(subset(tp, size == 2), subset(fp, size == 2))
p3 = calcPrecision(subset(tp, size == 3), subset(fp, size == 3))
p4 = calcPrecision(subset(tp, size == 4), subset(fp, size == 4))
p5 = calcPrecision(subset(tp, size == 5), subset(fp, size == 5))

avg_recall = calcRecall(tp, fn)
r1 = calcRecall(subset(tp, size == 1), subset(fn, size == 1))
r2 = calcRecall(subset(tp, size == 2), subset(fn, size == 2))
r3 = calcRecall(subset(tp, size == 3), subset(fn, size == 3))
r4 = calcRecall(subset(tp, size == 4), subset(fn, size == 4))
r5 = calcRecall(subset(tp, size == 5), subset(fn, size == 5))

size = c(1, 2, 3, 4, 5)
recall_s = c(r1, r2, r3, r4, r5)
precision_s = c(p1, p2, p3, p4, p5)

size_df = data.frame('precision' = precision_s, 'recall' = recall_s, 'size' = size)

ggplot(size_df, aes(x=size,y=precision)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_precision)
ggplot(size_df, aes(x=size,y=recall)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_recall)


p1 = calcPrecision(subset(tp, density_bins == 1), subset(fp, density_bins == 1))
p2 = calcPrecision(subset(tp, density_bins == 2), subset(fp, density_bins == 2))
p3 = calcPrecision(subset(tp, density_bins == 3), subset(fp, density_bins == 3))
p4 = calcPrecision(subset(tp, density_bins == 4), subset(fp, density_bins == 4))
p5 = calcPrecision(subset(tp, density_bins == 5), subset(fp, density_bins == 5))


r1 = calcRecall(subset(tp, density_bins == 1), subset(fn, density_bins == 1))
r2 = calcRecall(subset(tp, density_bins == 2), subset(fn, density_bins == 2))
r3 = calcRecall(subset(tp, density_bins == 3), subset(fn, density_bins == 3))
r4 = calcRecall(subset(tp, density_bins == 4), subset(fn, density_bins == 4))
r5 = calcRecall(subset(tp, density_bins == 5), subset(fn, density_bins == 5))

density = c(1, 2, 3, 4, 5)
recall_d = c(r1, r2, r3, r4, r5)
precision_d = c(p1, p2, p3, p4, p5)

density_df = data.frame('precision' = precision_d, 'recall' = recall_d, 'density' = density)

unique(density)

density_df = binnedPR(pred, obs, density, unique(density))

ggplot(density_df, aes(x=density,y=precision)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_precision)
ggplot(density_df, aes(x=density,y=recall)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_recall)
