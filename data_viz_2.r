library(ggplot2)
library(dplyr)

calcPrecision = function(tp, fp){
  precision = tp/(tp+fp)
  return(precision)
}

calcRecall = function(tp, fn){
  recall = tp/(tp+fn)
  return(recall)
}

binnedPR = function(pred_df, obs_df, target_col, bin_list){
  output_df = data.frame('bins' = bin_list, 'precision' = 0, 'recall' = 0)
  for (i in bin_list){
    current_df = subset(pred_df, target_col == i)
    current_obs = subset(obs_df, target_col == i)
    tp = nrow(subset(current_df, gt_uid > 0 & y_pred == 1))
    tn = nrow(subset(current_df, gt_uid == 0 & y_pred == 0))
    fp = nrow(subset(current_df, gt_uid == 0 & y_pred == 1))
    fn = nrow(subset(current_obs, gt_uid > 0 & y_pred == 0)) - nrow(subset(current_df, gt_uid > 0 & y_pred == 0))
    precision = calcPrecision(tp, fp)
    recall = calcRecall(tp, fn)
    output_df$precision[output_df$bins == i] = precision
    output_df$recall[output_df$bins == i] = recall
  }
  return(output_df)
}

pred_csv = '/Users/theo/data/hough_dataset/predicted_train_cleaned.csv'
ground_csv = '/Users/theo/data/hough_dataset/predicted_train.csv'

pred = read.csv(pred_csv)
obs = read.csv(ground_csv)

tp = subset(pred, gt_uid > 0 & y_pred == 1)
tn = subset(pred, gt_uid == 0 & y_pred == 0)
fp = subset(pred, gt_uid == 0 & y_pred == 1)
fn = subset(obs, gt_uid > 0 & y_pred == 0)

avg_precision = calcPrecision(nrow(tp), nrow(fp))
avg_recall = calcRecall(nrow(tp), nrow(fn))

pred$size = ntile(pred$r, 10)

size_df = binnedPR(pred, obs, pred$size, unique(pred$size))

plot_df = data.frame('precision' = size_df$precision, 'recall' = size_df$recall, 'plot' = as.factor(size_df$bins))

ggplot(plot_df, aes(x=plot,y=precision)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_precision) + ggtitle("Training set precision") + xlab("Size class") + ylab("Precision")
ggplot(plot_df, aes(x=plot,y=recall)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_recall) + ggtitle("Training set recall") + xlab("Size class") + ylab("Recall")

obs$density = 0
pred$density = 0
temp = subset(obs, gt_uid > 0, select=plot)
plots = as.data.frame(table(temp))
for (i in plots$Freq){
  current_plot = plots[plots$Freq == i, 1]
  pred$density[pred$plot == current_plot] = i
  obs$density[obs$plot == current_plot] = i
}

density_df = binnedPR(pred, obs, pred$density, unique(pred$density))

plot_df = data.frame('precision' = density_df$precision, 'recall' = density_df$recall, 'plot' = as.factor(density_df$bins))

ggplot(plot_df, aes(x=plot,y=precision)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_precision)
ggplot(plot_df, aes(x=plot,y=recall)) + geom_col() + ylim(0,1) + geom_hline(yintercept=avg_recall)


radius_est = subset(pred, y_pred == 1)
radius_est$oops = radius_est$gt_r_diff+radius_est$r
reg = lm(radius_est$r~radius_est$oops)
plot(radius_est$r, radius_est$oops, xlim=(c(0,1)), ylim=(c(0,1)))
abline(reg, col=rgb(1,0,0,1), lwd=2)
abline(a=0,b=1, col=rgb(0,0,0,1), lwd=2)        

ggplot(radius_est, aes(x=r)) + geom_histogram() + geom_hline(yintercept = mean(radius_est$gt_r_diff))
