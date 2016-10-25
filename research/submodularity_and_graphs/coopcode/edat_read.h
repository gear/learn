#ifndef __EDAT_READ_H__
#define __EDAT_READ_H__

#include "edge_data.h"

/*
  read in images and create an edge data object

  If you use this software for a publication, then cite the following papers:
  
  S. Jegelka and J. Bilmes. "Submodularity beyond submodular energies: coupling
  edges in graph cuts". IEEE Conference on Computer Vision and Pattern Recognition 
  (CVPR), 2011.
 */

class Edat_read: public Edge_data {

 public:
  Edat_read() { };
  ~Edat_read();

  // create edge weights from an input image
  void readWeightsFromImage(const char* imagename);
  // read in the edge classes from a binary file
  void readClasses(const char* classfile, double modthresh=0.0000000001);
  void readClasses(int* classes_in, int num_classes, double modthresh=0.0000000001);

  /* inherited:
     double* weights;  // array of edge weights
     int* classes;  // array of edge classes
     unsigned int m, nclass;
     int height, width, channels;
     // m is the number of edges
     // nclass is the number of classes
     // channels is the number of channels
     
     // weights has length m, and classes has length 2*m, since edges
     // (i,j) and (j,i) have teh same weights, but can have different
     // classes
   */

};

#endif
