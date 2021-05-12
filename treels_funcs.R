#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

require(TreeLS)

input = args[1]
map_path = args[2]
seg_path = args[3]
plot = args[4]

getTreeLSDetections = function(tls_path, xy_path, seg_path, h_step=1, max_h=3, pixel_size=0.01, n_ransac=10, viz=FALSE){
  tls = readTLS(tls_path)
  tls = tlsSample(tls, smp.voxelize(pixel_size))
  if (viz == TRUE) {x = plot(tls)}
  
  map = treeMap(tls, map.hough(h_step=h_step, max_h=max_h, pixel_size=pixel_size))
  
  if (viz == TRUE) {add_treeMap(x, map, color='red')}
  
  rads = map@data[Radii > 0, mean(Radii), by=TreeID]
  xymap = treeMap.positions(map)
  tls = treePoints(tls, map, trp.crop(circle=FALSE))
  tls = stemPoints(tls, stm.hough(pixel_size=pixel_size))
  seg = stemSegmentation(tls, sgt.ransac.circle(n = n_ransac))
  
  if (viz == TRUE) {tlsPlot(tls, seg, map)}
  
  write.csv(as.data.frame(xymap), file=xy_path)
  write.csv(as.data.frame(seg), file=seg_path)
}

getTreeLSDetections(input, map_path, seg_path, viz=plot, pixel_size=0.2)
