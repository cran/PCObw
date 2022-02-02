#include <Rcpp.h>
#include <algorithm>
#include <RcppEigen.h>


using namespace Rcpp;
//[[Rcpp::depends(RcppEigen)]]
using Eigen::MatrixXd;
using Eigen::Map;
using Eigen::VectorXd;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrixXd;




/********************************************************************************************/
/**                                           1D                                           **/
/********************************************************************************************/



class criterion_1D{
  
public:
  Eigen::VectorXd xi; // initialised in criterion_1D constructor
  double hmin; // initialised in XK_exact_crit_1D constructor or XK_binned_crit_1D constructor
  
protected:
  int n; // initialised in criterion_1D constructor
  double n2;  // initialised in criterion_1D constructor
  double hmin2; // initialised in XK_exact_crit_1D constructor or XK_binned_crit_1D constructor
  
public:
  /**
   * Constructor of criterion_1D
   * 
   * @param xi vector of the n observations
   */
  criterion_1D(Eigen::VectorXd xi = Eigen::VectorXd::Zero(1)) {
    this->xi = xi;
    this->n = xi.size();
    this->n2 = n * n;
  }
  
public:
  virtual Eigen::VectorXd compute(Eigen::ArrayXd H) = 0;
  
  virtual double compute(double H) = 0;
  
};




class exact_crit_1D : public criterion_1D {
protected :
  Eigen::ArrayXd u;
  
public :
  exact_crit_1D(Eigen::VectorXd xi):criterion_1D(xi){
    
  }
  
  
};



class binned_crit_1D : public criterion_1D {
  
public:
  int nb_bin;
  Eigen::VectorXi tabulated;
  
  
protected:
  double delta;
  Eigen::VectorXd bin_cnt;
  
public :
  binned_crit_1D(Eigen::VectorXd xi, int nb_bin):criterion_1D(xi){
    compute_tab(nb_bin);
  }
  
  
public:
  /*
   * updates nb_bin and delta
   * computes the number of observations in each bin
   */
  void compute_tab(int nb){
    this->nb_bin = nb;
    this->delta = (xi.maxCoeff() - xi.minCoeff()) * 1.01 / nb;
    
    Eigen::VectorXd xx = xi / delta;
    Eigen::VectorXi x = xx.cast <int> ();
    x = x.array() - x.minCoeff() + 1;
    std::sort(x.data(), x.data() + x.size());
    
    // copy of x
    std::vector<int> x_c(&x(0), x.data() + x.size());
    
    // ATTENTION std::unique modifies x_c
    std::vector<int>::iterator it_M = std::unique(x_c.begin(), x_c.end());
    
    // counts occurences of each element of x
    --it_M;
    tabulated = Eigen::VectorXi::Zero(nb);
    int m;
    while (it_M != x_c.begin() - 1){
      m = *it_M;
      tabulated(m - 1) = (x.array() == m).count();
      --it_M;
    }
  }
  
public:
  
  // digonal terms excluded
  void f_bin_cnt_diagout(){
    bin_cnt = Eigen::VectorXd::Zero(nb_bin);
    for (int ii = 0; ii < nb_bin; ii++){
      int w = tabulated(ii);
      bin_cnt(0) += w * (w - 1.);
      for (int jj = 0; jj < ii; jj++){
        bin_cnt(ii - jj) += w * tabulated(jj);
      }
    }
    bin_cnt(0) *= 0.5;
  }
  
  // diagonal terms included
  void f_bin_cnt_diagin(){
    bin_cnt = Eigen::VectorXd::Zero(nb_bin);
    for (int ii = 0; ii < nb_bin; ii++){
      int w = tabulated(ii);
      bin_cnt(0) += w * w;
      for (int jj = 0; jj < ii; jj++){
        bin_cnt(ii - jj) += w * tabulated(jj);
      }
    }
    bin_cnt(0) *= 0.5;
  }
  
  
};




/********************************************************************************************/
/**                               1D exact for 3 kernels                                   **/
/********************************************************************************************/

/*
 * The class of exact 1D criterion with Gaussian kernel
 */
class GK_exact_crit_1D : public exact_crit_1D{
private :
  Eigen::VectorXd k;
  double pen_factor;
  
  
public :
  GK_exact_crit_1D(Eigen::VectorXd xi) : exact_crit_1D(xi){
    this->hmin = M_1_SQRT_2PI / double(n);
    this->hmin2 = hmin * hmin;
    this->pen_factor = - (M_LN2 + std::log(double(n)) + M_LN_SQRT_PI);
  }
  
public:
  Eigen::VectorXd compute(Eigen::ArrayXd H){
    
    
    Eigen::ArrayXd Hlog = H.log();
    Eigen::ArrayXd H2 = H.square();
    
    Eigen::ArrayXd H2_p_hmin2 = H2 + hmin2;
    Eigen::VectorXd pen = (pen_factor - Hlog).exp();
    
    Eigen::VectorXd loss(H.size());
    
    Eigen::ArrayXd c1 = -(M_LN2 + M_LN_SQRT_PI + Hlog);
    Eigen::ArrayXd c2 = -(M_LN_SQRT_2PI + 0.5 * H2_p_hmin2.log());
    
    Eigen::ArrayXd ch_1 = - 0.25 / H2;
    Eigen::ArrayXd ch_2 = - 0.5 / H2_p_hmin2;
    
    int n_u = n - 1;
    
    Eigen::VectorXd somme_1 = Eigen::VectorXd::Zero(H.size());
    Eigen::VectorXd somme_2 = Eigen::VectorXd::Zero(H.size());
    
    for (int i = 0; i < (n - 1); i++){
      Rcpp::checkUserInterrupt();
      u = (xi.segment(i + 1, n_u).array() - xi(i)).square();
      for (int no_h = 0; no_h < H.size(); no_h++){
        
        k = (c1(no_h) + ch_1(no_h) * u).exp();
        somme_1(no_h) += k.sum();
        
        k = (c2(no_h) + ch_2(no_h) * u).exp();
        somme_2(no_h) += k.sum();
        
      }
      n_u -= 1;
    }
    loss = 2 * somme_1 - 4 * somme_2;
    loss /= n2;
    Eigen::VectorXd crit = loss + pen;
    return crit;
  }
  
  
  
  double compute(double h){
    double Hlog = std::log(h);
    double H2 = h * h;
    
    double H2_p_hmin2 = H2 + hmin2;
    double pen = std::exp(pen_factor - Hlog);
    
    double loss;
    
    double c1 = -(M_LN2 + M_LN_SQRT_PI + Hlog);
    double c2 = -(M_LN_SQRT_2PI + 0.5 * std::log(H2_p_hmin2));
    
    double ch_1 = - 0.25 / H2;
    double ch_2 = - 0.5 / H2_p_hmin2;
    
    int n_u = n - 1;
    
    double somme_1 = 0;
    double somme_2 = 0;
    
    for (int i = 0; i < (n - 1); i++){
      
      Rcpp::checkUserInterrupt();
      
      u = (xi.segment(i + 1, n_u).array() - xi(i)).square();
      
      k = (c1 + ch_1 * u).exp();
      somme_1 += k.sum();

      k = (c2 + ch_2 * u).exp();
      somme_2 += k.sum();
      
      n_u -= 1;
    }
    loss = 2 * somme_1 - 4 * somme_2;

    loss = loss / n2;
    double crit = loss + pen;
    return crit;
  }
  
  
};

/*
 * The class of exact 1D criterion with Epanechnikov kernel
 */
class EK_exact_crit_1D : public exact_crit_1D{
public :
  EK_exact_crit_1D(Eigen::VectorXd xi) : exact_crit_1D(xi){
    this->hmin = 0.75 / double(n);
    this->hmin2 = hmin * hmin;
    std::sort(this->xi.data(), this->xi.data() + this->xi.size());
  }
  
  
  
public:
  Eigen::VectorXd compute(Eigen::ArrayXd H){
    Eigen::VectorXd pen = 0.6 / (H * double(n));//0.6 = 3/5 //1.5 * (H.square() - hmin2 / 5.0) * H.pow(-3) / double(n);
    
    Eigen::VectorXd loss = Eigen::VectorXd::Zero(H.size());
    int n_u = n - 1;
    
    
    double h, hinv, itu, r, r2, c_k1_0, c_k1_2, c_k1_3, c_k1_5, c_k2_0, c_k2_2;
    int u_ind;
    double z, z2, k1, k2_t_2;
    double c_k3, k3_p1, k3_p2;
    
    double b1, b2, b3;
    
    
    for (int i = 0; i < (n - 1); i++){
      // since xi is sorted in increasing order, the elemnts of u are >=0
      // exept near 0 because of numerical precision
      // and u is also already sorted in increasing order
      u = (xi.segment(i + 1, n_u).array() - xi(i)).abs();
      
      
      for (int no_h = 0; no_h < H.size(); no_h++){
        Rcpp::checkUserInterrupt();
        h = H(no_h);
        hinv = 1 / h;
        // index for loop on u = (oi - oj)
        u_ind = 0;
        itu = u(u_ind);
        
        r = hmin * hinv;
        r2 = r * r;
        
        
        // multiplicative constant
        
        c_k1_0 = 1.2 * hinv; // 32 * 2 * 3 / (160 * h)
        c_k1_2 = -1.5 * hinv; // 40 * 2 * 3 / (160 * h)
        c_k1_3 = 0.75 * hinv;// 20 * 2 * 3 / (160 * h)
        c_k1_5 = -0.0375 * hinv; // 2 * 3 / (160 * h)
        
        c_k2_2 = 3 * hinv; // 2 * 5 * 2 * 3 / (20 * h)
        c_k2_0 = c_k2_2 - 0.6 * r2 * hinv; // 2 * 2 * 3 * (5 - r2) / (20 * h);
        
        c_k3 = 0.075 * hinv / (r2 * r);//0.075 = 2 * 2 * 3 / 160
        
        b1 = h - hmin;
        b2 = h + hmin;
        b3 = h + h;
        
        
        while ((itu <  b1)&&(u_ind < u.size())){// itu is in [0; h - hmin[
          
          // (Kh * Kh) (itu)
          z = itu * hinv;
          z2 = z * z;
          
          k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
          
          
          // (Kh * Khmin) (itu)
          k2_t_2 = c_k2_0 - c_k2_2 * z2; //0.3 * (5 - r2 - 5 * z2) * hinv; // 0.3 = 2*3/20
          
          
          loss(no_h) += k1 - k2_t_2;
          u_ind++;
          itu = u(u_ind);
          
        }
        while ((itu <  b2)&&(u_ind < u.size())){// itu is in [h - hmin; h + hmin[
          
          // (Kh * Kh) (itu)
          z = itu * hinv;
          z2 = z * z;
          
          k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
          
          // (Kh * Khmin) (itu)
          k3_p1 = std::pow(1 + r - z, 3);
          k3_p2 = 4 * (1 + r2) - 3 * (r * (z + 4) + z) - z2;
          
          k2_t_2 = - c_k3 * k3_p1 * k3_p2;
          
          loss(no_h) += k1 - k2_t_2;
          u_ind++;
          itu = u(u_ind);
          
        }
        while ((itu <  b3)&&(u_ind < u.size())){// itu is in [h + hmin; 2h[
          
          // (Kh * Kh) (itu)
          z = itu * hinv;
          z2 = z * z;
          
          k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
          
          // (Kh * Khmin) (itu) = 0
          
          loss(no_h) += k1;
          u_ind++;
          itu = u(u_ind);
          
        }
        
      }
      n_u -= 1;
    }
    loss /= n2;
    
    
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  double compute(double h){
    double pen = 0.6 / (h * double(n));//0.6 = 3/5 //1.5 * (H.square() - hmin2 / 5.0) * H.pow(-3) / double(n);
    
    double loss = 0;
    int n_u = n - 1;
    
    
    double hinv, itu, r, r2, c_k1_0, c_k1_2, c_k1_3, c_k1_5, c_k2_0, c_k2_2;
    int u_ind;
    double z, z2, k1, k2_t_2;
    // double c_k3_0, c_k3_1, c_k3_2, c_k3_3, c_k3_5;
    double c_k3, k3_p1, k3_p2;
    
    double b1, b2, b3;
    
    
    for (int i = 0; i < (n - 1); i++){
      // since xi is sorted in increasing order, the elemnts of u are >=0
      // exept near 0 because of numerical precision
      // and u is also already sorted in increasing order
      u = (xi.segment(i + 1, n_u).array() - xi(i)).abs();
      
      
      
      Rcpp::checkUserInterrupt();
      
      hinv = 1 / h;
      // index for loop on u = (oi - oj)
      u_ind = 0;
      itu = u(u_ind);
      
      r = hmin * hinv;
      r2 = r * r;
      
      
      // multiplicative constant
      
      c_k1_0 = 1.2 * hinv; // 32 * 2 * 3 / (160 * h)
      c_k1_2 = -1.5 * hinv; // 40 * 2 * 3 / (160 * h)
      c_k1_3 = 0.75 * hinv;// 20 * 2 * 3 / (160 * h)
      c_k1_5 = -0.0375 * hinv; // 2 * 3 / (160 * h)
      
      c_k2_2 = 3 * hinv; // 2 * 5 * 2 * 3 / (20 * h)
      c_k2_0 = c_k2_2 - 0.6 * r2 * hinv; // 2 * 2 * 3 * (5 - r2) / (20 * h);
      
      
      
      c_k3 = 0.075 * hinv / (r2 * r);//0.075 = 2 * 2 * 3 / 160
      
      b1 = h - hmin;
      b2 = h + hmin;
      b3 = h + h;
      
      
      while ((itu <  b1)&&(u_ind < u.size())){// itu is in [0; h - hmin[
        
        // (Kh * Kh) (itu)
        z = itu * hinv;
        z2 = z * z;
        
        k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
        
        
        // (Kh * Khmin) (itu)
        k2_t_2 = c_k2_0 - c_k2_2 * z2; //0.3 * (5 - r2 - 5 * z2) * hinv; // 0.3 = 2*3/20
        
        
        loss += k1 - k2_t_2;
        u_ind++;
        itu = u(u_ind);
        
      }
      while ((itu <  b2)&&(u_ind < u.size())){// itu is in [h - hmin; h + hmin[
        
        // (Kh * Kh) (itu)
        z = itu * hinv;
        z2 = z * z;
        
        k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
        
        // (Kh * Khmin) (itu)
        k3_p1 = std::pow(1 + r - z, 3);
        k3_p2 = 4 * (1 + r2) - 3 * (r * (z + 4) + z) - z2;
        
        k2_t_2 = - c_k3 * k3_p1 * k3_p2;
        
        loss += k1 - k2_t_2;
        u_ind++;
        itu = u(u_ind);
        
      }
      while ((itu <  b3)&&(u_ind < u.size())){// itu is in [h + hmin; 2h[
        
        // (Kh * Kh) (itu)
        z = itu * hinv;
        z2 = z * z;
        
        k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
        
        // (Kh * Khmin) (itu) = 0
        
        loss += k1;
        u_ind++;
        itu = u(u_ind);
        
      }
      
      n_u -= 1;
    }
    loss /= n2;
    
    
    double crit = loss + pen;
    return(crit);
  }
  
};

/*
 * The class of exact 1D criterion with Biweight kernel
 */
class BK_exact_crit_1D : public exact_crit_1D{
public :
  BK_exact_crit_1D(Eigen::VectorXd xi) : exact_crit_1D(xi){
    this->hmin = 15.0 / (16.0 * double(n));
    this->hmin2 = hmin * hmin;
    std::sort(this->xi.data(), this->xi.data() + this->xi.size());
  }
  
public:
  Eigen::VectorXd compute(Eigen::ArrayXd H){
    // Eigen::ArrayXd H_inv = H.inverse();
    
    // Eigen::VectorXd pen =  5.0 * H_inv / (7.0 * double(n));
    Eigen::VectorXd pen =  5.0 * H.inverse() / (7.0 * double(n));
    
    Eigen::VectorXd loss = Eigen::VectorXd::Zero(H.size());
    double h, h_inv, r, r2, itu, z, z2, k1, k2_t_2;
    double c_k1_0, c_k1_2, c_k1_4, c_k1_5, c_k1_7, c_k1_9, c_k2_0, c_k2_2, c_k2_4;
    double k3_p1, k3_p2, c_k3, c_k3_p2_0, c_k3_p2_1, c_k3_p2_2, c_k3_p2_3;
    double hinv_t_15,  b1, b2, b3;
    int u_ind;
    int n_u = n - 1;
    
    
    
    for (int i = 0; i < (n - 1); i++){
      // since xi is sorted in increasing order, the elemnts of u are >=0 and sorted in increasing order
      // exept near 0 because of numerical precision u elements can be < 0
      u = (xi.segment(i + 1, n_u).array() - xi(i)).abs();
      
      for (int no_h = 0; no_h < H.size(); no_h++){
        Rcpp::checkUserInterrupt();
        h = H(no_h);
        h_inv = 1 / h;
        
        // r = hmin / h;
        r = hmin * h_inv;
        r2 = r * r;
        // r5 = r2 * r2 * r;
        
        hinv_t_15 = 15 * h_inv;
        c_k1_9 = - h_inv / 358.4;
        c_k1_7 = hinv_t_15 / 224.0;
        c_k1_5 = - hinv_t_15 / 16.0;
        c_k1_4 = hinv_t_15 / 8.0;
        c_k1_2 = - hinv_t_15 / 7.0;
        c_k1_0 = h_inv / 0.7;
        
        c_k2_0 = h_inv * (105 + r2 * (5 * r2 - 30)) / 28.0;
        c_k2_2 = h_inv * (45 * r2 - 105) / 14.0;
        c_k2_4 = 3.75 * h_inv;
        
        c_k3 = 20 * h_inv / (3584.0 * r * r2 * r2);
        
        c_k3_p2_0 = 16 * (1 + r * (r - 1) * (5 + r * (r - 4)));
        c_k3_p2_1 = -5 * (1 + r) * (5 + r * (5 * r - 14));
        c_k3_p2_2 = 3 * (1 + r * (10 + r));
        c_k3_p2_3 = 5 * (1 + r);
        
        u_ind = 0;
        itu = u(u_ind);
        
        b1 = h - hmin;
        b2 = h + hmin;
        b3 = h + h;
        
        while ((itu <  b1) && (u_ind < u.size())){// itu is in [0; h - hmin[
          z = itu * h_inv;
          z2 = z * z;
          
          // (Kh * Kh) (itu)
          k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
          
          // (Kh * Khmin) (itu)
          k2_t_2 = c_k2_0 + z2 * (c_k2_2 + z2 * c_k2_4);
          
          loss(no_h) += k1 - k2_t_2;
          
          u_ind++;
          itu = u(u_ind);
        }
        while ((itu <  b2) && (u_ind < u.size())) {// itu is in [h - hmin; h + hmin[
          z = itu * h_inv;
          z2 = z * z;
          
          // (Kh * Kh) (itu)
          k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
          
          // (Kh * Khmin) (itu)
          k3_p1 = std::pow(1 + r - z, 5);
          k3_p2 = c_k3_p2_0 + z * (c_k3_p2_1 + z * (c_k3_p2_2 + z * (c_k3_p2_3 + z)));
          
          k2_t_2 = c_k3 * k3_p1 * k3_p2;
          
          loss(no_h) += k1 - k2_t_2;
          
          u_ind++;
          itu = u(u_ind);
        }
        while ((itu <  b3) && (u_ind < u.size())){// itu is in [h + hmin; 2 * h[
          z = itu * h_inv;
          z2 = z * z;
          
          // (Kh * Kh) (itu)
          k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
          
          // (Kh * Khmin) (itu) = 0
          
          loss(no_h) += k1;
          u_ind++;
          itu = u(u_ind);
        }
      }
      n_u -= 1;
    }
    loss /= n2;
    
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  double compute(double h){
    double h_inv = 1 / h;
    
    double pen =  5.0 * h_inv / (7.0 * double(n));
    
    double loss = 0;
    double r, r2, r3, r5, itu, z, z2, k1, k2_t_2;
    double c_k1_0, c_k1_2, c_k1_4, c_k1_5, c_k1_7, c_k1_9, c_k2_0, c_k2_2, c_k2_4;
    
    double k3_p1, k3_p2, c_k3, c_k3_p2_0, c_k3_p2_1, c_k3_p2_2, c_k3_p2_3;
    double hinv_t_15,  b1, b2, b3;
    int u_ind;
    int n_u = n - 1;
    
    
    
    for (int i = 0; i < (n - 1); i++){
      // since xi is sorted in increasing order, the elemnts of u are >=0 and sorted in increasing order
      // exept near 0 because of numerical precision u elements can be < 0
      u = (xi.segment(i + 1, n_u).array() - xi(i)).abs();
      
      
      Rcpp::checkUserInterrupt();
      
      
      
      r = hmin * h_inv;
      r2 = r * r;
      r3 = r2 * r;
      
      r5 = r3 * r2;
      
      hinv_t_15 = 15 * h_inv;
      c_k1_9 = - h_inv / 358.4;
      c_k1_7 = hinv_t_15 / 224.0;
      c_k1_5 = - hinv_t_15 / 16.0;
      c_k1_4 = hinv_t_15 / 8.0;
      c_k1_2 = - hinv_t_15 / 7.0;
      c_k1_0 = h_inv / 0.7;
      
      
      
      
      c_k2_0 = h_inv * (105 + r2 * (5 * r2 - 30)) / 28.0;
      c_k2_2 = h_inv * (45 * r2 - 105) / 14.0;
      c_k2_4 = 3.75 * h_inv;
      
      c_k3 = 20 * h_inv / (3584.0 * r5);
      
      
      
      c_k3_p2_0 = 16 * (1 + r * (r - 1) * (5 + r * (r - 4)));
      c_k3_p2_1 = -5 * (1 + r) * (5 + r * (5 * r - 14));
      c_k3_p2_2 = 3 * (1 + r * (10 + r));
      c_k3_p2_3 = 5 * (1 + r);
      
      u_ind = 0;
      itu = u(u_ind);
      
      b1 = h - hmin;
      b2 = h + hmin;
      b3 = h + h;
      
      while ((itu <  b1) && (u_ind < u.size())){// itu is in [0; h - hmin[
        // (Kh * Kh) (itu)
        
        z = itu * h_inv;
        z2 = z * z;
        
        k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
        
        
        // (Kh * Khmin) (itu)
        
        k2_t_2 = c_k2_0 + z2 * (c_k2_2 + z2 * c_k2_4);
        
        
        
        loss += k1 - k2_t_2;
        
        u_ind++;
        itu = u(u_ind);
      }
      while ((itu <  b2) && (u_ind < u.size())) {// itu is in [h - hmin; h + hmin[
        // (Kh * Kh) (itu)
        z = itu * h_inv;
        z2 = z * z;
        
        k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
        
        // (Kh * Khmin) (itu)
        k3_p1 = std::pow(1 + r - z, 5);
        k3_p2 = c_k3_p2_0 + z * (c_k3_p2_1 + z * (c_k3_p2_2 + z * (c_k3_p2_3 + z)));
        
        
        k2_t_2 = c_k3 * k3_p1 * k3_p2;
        
        loss += k1 - k2_t_2;
        
        u_ind++;
        itu = u(u_ind);
      }
      while ((itu <  b3) && (u_ind < u.size())){// itu is in [h + hmin; 2 * h[
        // (Kh * Kh) (itu)
        z = itu * h_inv;
        z2 = z * z;
        
        k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
        
        
        // (Kh * Khmin) (itu) = 0
        loss += k1;
        u_ind++;
        itu = u(u_ind);
      }
      
      n_u -= 1;
    }
    loss /= n2;
    
    double crit = loss + pen;
    return(crit);
  }
  
  
};



/********************************************************************************************/
/**                               1D binned for 3 kernels                                  **/
/********************************************************************************************/

/*
 * The class of binned 1D criterion with Gaussian kernel
 */
class GK_binned_crit_1D : public binned_crit_1D{
  
public :
  GK_binned_crit_1D(Eigen::VectorXd xi, int nb_bin) : binned_crit_1D(xi, nb_bin){
    this->hmin = M_1_SQRT_2PI / double(n);
    this->hmin2 = hmin * hmin;
    f_bin_cnt_diagout();
  }
  
  
  
public:
  Eigen::VectorXd compute(Eigen::ArrayXd H){
    Eigen::VectorXd pen = 1 / (2.0 * M_SQRT_PI * H * double(n));
    Eigen::VectorXd loss(H.size());
    double h, h2, sum, h_1, h_2;
    int l1, l2, nb;
    double d1, d2, t1, t2, term;
    for (int no_h=0; no_h < H.size(); no_h++){
      
      h = H(no_h);
      h2 = h * h;
      
      sum = 0.0;
      
      h_1 = M_SQRT2 * h;
      l1 = std::min(nb_bin - 1, int(std::ceil(5 * h_1 / delta)));
      h_2 = std::sqrt(h2 + hmin2);
      l2 = std::min(nb_bin - 1, int(std::ceil(5 * h_2 / delta)));
      
      nb = std::max(l1, l2);
      
      for (int i = 0; i < nb; i++){
        
        d1 = i * delta / h;
        d1 *= d1;
        d2 = i * delta / h_2;
        d2 *= d2;
        
        t1 = std::exp(-d1 / 4.0) / h;
        t2 = std::sqrt(8.0 / (h2 + hmin2)) * std::exp(-d2 / 2.0);
        
        term = t1 - t2;
        sum += term * bin_cnt[i];
      }
      loss(no_h) = sum / (n2 * M_SQRT_PI);
    }
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  
  double compute(double h){
    double pen = 1 / (2.0 * M_SQRT_PI * h * double(n));
    
    double loss;
    double h2, sum, h_1, h_2;
    int l1, l2, nb;
    double d1, d2, t1, t2, term;
    
    h2 = h * h;
    
    sum = 0.0;
    
    h_1 = M_SQRT2 * h;
    l1 = std::min(nb_bin - 1, int(std::ceil(5 * h_1 / delta)));
    h_2 = std::sqrt(h2 + hmin2);
    l2 = std::min(nb_bin - 1, int(std::ceil(5 * h_2 / delta)));
    
    nb = std::max(l1, l2);
    
    
    for (int i = 0; i < nb; i++){
      
      d1 = i * delta / h;
      d1 *= d1;
      d2 = i * delta / h_2;
      d2 *= d2;
      
      t1 = std::exp(-d1 / 4.0) / h;
      t2 = std::sqrt(8.0 / (h2 + hmin2)) * std::exp(-d2 / 2.0);
      
      term = t1 - t2;
      sum += term * bin_cnt[i];
    }
    loss = sum / (n2 * M_SQRT_PI);
    
    
    
    double crit = loss + pen;
    
    
    return(crit);
  }
  
};

/*
 * The class of binned 1D criterion with Epanechnikov kernel
 */
class EK_binned_crit_1D : public binned_crit_1D{
  
public :
  EK_binned_crit_1D(Eigen::VectorXd xi, int nb_bin) : binned_crit_1D(xi, nb_bin){
    this->hmin = 0.75 / double(n);
    this->hmin2 = hmin * hmin;
    f_bin_cnt_diagout();
  }
  
  
  
public:
  Eigen::VectorXd compute(Eigen::ArrayXd H){
    Eigen::VectorXd pen = 0.6 / (H * double(n));//1.5 * (H.square() - hmin2 / 5.0) * H.pow(-3) / double(n);
    
    Eigen::VectorXd loss = Eigen::VectorXd::Zero(H.size());
    double h, hinv, sum, ratio, r, r2, z, z2, k1, k2_t_2;
    int i, nb, b_1, b_2, b_3;
    
    
    double c_k1_0, c_k1_2, c_k1_3, c_k1_5, c_k2_2, c_k2_0;
    double c_k3, k3_p1, k3_p2;
    
    
    
    nb = bin_cnt.size();
    
    for (int no_h=0; no_h < H.size(); no_h++){
      h = H(no_h);
      hinv = 1 / h;
      
      sum = 0.0;
      
      i = 0;
      
      b_1 = std::min(nb, int((h - hmin) / delta) + 1);
      b_2 = std::min(nb, int((h + hmin) / delta) + 1);
      b_3 = std::min(nb, int((2 * h) / delta) + 1);
      
      ratio = delta * hinv;
      
      r = hmin * hinv;
      r2 = r * r;
      
      
      // multiplicative constant
      c_k1_0 = 1.2 * hinv; // 32 * 2 * 3 / (160 * h)
      c_k1_2 = -1.5 * hinv; // 40 * 2 * 3 / (160 * h)
      c_k1_3 = 0.75 * hinv;// 20 * 2 * 3 / (160 * h)
      c_k1_5 = -0.0375 * hinv; // 2 * 3 / (160 * h)
      
      c_k2_2 = 3 * hinv; // 2 * 5 * 2 * 3 / (20 * h)
      c_k2_0 = c_k2_2 - 0.6 * r2 * hinv; // 2 * 2 * 3 * (5 - r2) / (20 * h);
      
      c_k3 = 0.075 * hinv / (r2 * r);//0.075 = 2 * 2 * 3 / 160
      
      
      while (i < b_1){// i * delta is in [0, h - hm[
        z = i * ratio;
        z2 = z * z;
        
        // (Kh \star Kh) (i*delta/h)
        k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
        
        // (Kh \star Khmin) (i*delta/h)
        k2_t_2 = c_k2_0 - c_k2_2 * z2;
        
        sum += (k1 - k2_t_2) * bin_cnt(i);
        i++;
      }
      while (i < b_2){// i * delta is in [h - hm, h + hm[
        z = i * ratio;
        z2 = z * z;
        
        // (Kh \star Kh) (i*delta/h)
        k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
        
        // (Kh \star Khmin) (i*delta/h)
        k3_p1 = std::pow(1 + r - z, 3);
        k3_p2 = 4 * (1 + r2) - 3 * (r * (z + 4) + z) - z2;
        
        k2_t_2 = - c_k3 * k3_p1 * k3_p2;
        
        sum += (k1 - k2_t_2) * bin_cnt(i);
        i++;
      }
      while (i < b_3){// i * delta is in [h + hm, 2h[
        z = i * ratio;
        z2 = z * z;
        
        // (Kh \star Kh) (i*delta/h)
        k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
        
        // (Kh \star Khmin) (i*delta/h) = 0
        
        sum += k1 * bin_cnt(i);
        i++;
      }
      loss(no_h) = sum;
    }
    loss /= n2;
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  
  double compute(double h){
    double pen = 0.6 / (h * double(n));//1.5 * (H.square() - hmin2 / 5.0) * H.pow(-3) / double(n);
    
    double loss = 0;
    double hinv, sum, ratio, r, r2, z, z2, k1, k2_t_2;
    int i, nb, b_1, b_2, b_3;
    
    
    double c_k1_0, c_k1_2, c_k1_3, c_k1_5, c_k2_2, c_k2_0;
    double c_k3, k3_p1, k3_p2;
    
    
    
    nb = bin_cnt.size();
    
    
    hinv = 1 / h;
    
    
    sum = 0.0;
    
    
    
    i = 0;
    
    
    b_1 = std::min(nb, int((h - hmin) / delta) + 1);
    b_2 = std::min(nb, int((h + hmin) / delta) + 1);
    b_3 = std::min(nb, int((2 * h) / delta) + 1);
    
    
    ratio = delta * hinv;
    
    
    r = hmin * hinv;
    r2 = r * r;
    
    
    // multiplicative constant
    c_k1_0 = 1.2 * hinv; // 32 * 2 * 3 / (160 * h)
    c_k1_2 = -1.5 * hinv; // 40 * 2 * 3 / (160 * h)
    c_k1_3 = 0.75 * hinv;// 20 * 2 * 3 / (160 * h)
    c_k1_5 = -0.0375 * hinv; // 2 * 3 / (160 * h)
    
    c_k2_2 = 3 * hinv; // 2 * 5 * 2 * 3 / (20 * h)
    c_k2_0 = c_k2_2 - 0.6 * r2 * hinv; // 2 * 2 * 3 * (5 - r2) / (20 * h);
    
    
    
    c_k3 = 0.075 * hinv / (r2 * r);//0.075 = 2 * 2 * 3 / 160
    
    
    while (i < b_1){
      z = i * ratio;
      z2 = z * z;
      
      
      // (Kh * Kh) (itu)
      k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
      
      // (Kh * Khmin) (itu)
      k2_t_2 = c_k2_0 - c_k2_2 * z2;
      
      sum += (k1 - k2_t_2) * bin_cnt(i);
      i++;
    }
    while (i < b_2){
      z = i * ratio;
      z2 = z * z;
      
      
      // (Kh * Kh) (itu)
      k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
      
      // (Kh * Khmin) (itu)
      
      k3_p1 = std::pow(1 + r - z, 3);
      k3_p2 = 4 * (1 + r2) - 3 * (r * (z + 4) + z) - z2;
      
      k2_t_2 = - c_k3 * k3_p1 * k3_p2;
      
      sum += (k1 - k2_t_2) * bin_cnt(i);
      i++;
    }
    while (i < b_3){
      z = i * ratio;
      z2 = z * z;
      
      // // (Kh \star Kh) (i*delta/h)
      
      k1 = c_k1_0 + z2 * (c_k1_2 + z * (c_k1_3 + z2 * c_k1_5));
      // (Kh \star Khmin) (i*delta/h) = 0
      sum += k1 * bin_cnt(i);
      i++;
    }
    loss = sum;
    
    loss /= n2;
    double crit = loss + pen;
    return(crit);
  }
  
};

/*
 * The class of binned 1D criterion with Biweight kernel
 */
class BK_binned_crit_1D : public binned_crit_1D{
  
public :
  BK_binned_crit_1D(Eigen::VectorXd xi, int nb_bin) : binned_crit_1D(xi, nb_bin){
    this->hmin = 15.0 / (16.0 * double(n));
    this->hmin2 = hmin * hmin;
    f_bin_cnt_diagout();
  }
  
  
  
public:
  Eigen::VectorXd compute(Eigen::ArrayXd H){
    int i, nb, b_1, b_2, b_3;
    double h, h_inv, ratio, r, r2, z, z2, k1, k2_t_2;
    double hinv_t_15, c_k1_0, c_k1_2, c_k1_4, c_k1_5, c_k1_7, c_k1_9, c_k2_0, c_k2_2, c_k2_4;
    double k3_p1, k3_p2, c_k3, c_k3_p2_0, c_k3_p2_1, c_k3_p2_2, c_k3_p2_3;
    
    nb = bin_cnt.size();
    Eigen::VectorXd pen =  5.0 / (7.0 * H * double(n));
    
    Eigen::VectorXd loss = Eigen::VectorXd::Zero(H.size());
    for (int no_h = 0; no_h < H.size(); no_h++){
      Rcpp::checkUserInterrupt();
      h = H(no_h);
      h_inv = 1 / h;
      
      b_1 = std::min(nb, int((h - hmin) / delta) + 1);
      b_2 = std::min(nb, int((h + hmin) / delta) + 1);
      b_3 = std::min(nb, int((2 * h) / delta) + 1);
      
      ratio = delta * h_inv;
      
      r = hmin * h_inv;
      r2 = r * r;
      
      
      hinv_t_15 = 15 * h_inv;
      c_k1_9 = - h_inv / 358.4;
      c_k1_7 = hinv_t_15 / 224.0;
      c_k1_5 = - hinv_t_15 / 16.0;
      c_k1_4 = hinv_t_15 / 8.0;
      c_k1_2 = - hinv_t_15 / 7.0;
      c_k1_0 = h_inv / 0.7;
      
      c_k2_0 = h_inv * (105 + r2 * (5 * r2 - 30)) / 28.0;
      c_k2_2 = h_inv * (45 * r2 - 105) / 14.0;
      c_k2_4 = 3.75 * h_inv;
      
      c_k3 = 20 * h_inv / (3584.0 * r * r2 * r2);
      
      c_k3_p2_0 = 16 * (1 + r * (r - 1) * (5 + r * (r - 4)));
      c_k3_p2_1 = -5 * (1 + r) * (5 + r * (5 * r - 14));
      c_k3_p2_2 = 3 * (1 + r * (10 + r));
      c_k3_p2_3 = 5 * (1 + r);
      
      i = 0;
      
      
      while (i <  b_1){// i * delta is in [0, h - hm[
        z = i * ratio;
        z2 = z * z;
        
        // (Kh \star Kh) (i*delta/h)
        k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
        
        // (Kh \star Khmin) (i*delta/h)
        k2_t_2 = c_k2_0 + z2 * (c_k2_2 + z2 * c_k2_4);
        
        loss(no_h) += (k1 - k2_t_2) * bin_cnt(i);
        i++;
      }
      while (i <  b_2){// i * delta is in [h - hm, h + hm[
        
        z = i * ratio;
        z2 = z * z;
        
        // (Kh \star Kh) (i*delta/h)
        k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
        
        // (Kh \star Khmin) (i*delta/h)
        k3_p1 = std::pow(1 + r - z, 5);
        k3_p2 = c_k3_p2_0 + z * (c_k3_p2_1 + z * (c_k3_p2_2 + z * (c_k3_p2_3 + z)));
        
        k2_t_2 = c_k3 * k3_p1 * k3_p2;
        
        loss(no_h) += (k1 - k2_t_2) * bin_cnt(i);
        i++;
      }
      while (i <  b_3){// i * delta is in [h + hm, 2h[
        z = i * ratio;
        z2 = z * z;
        
        // (Kh \star Kh) (i*delta/h)
        k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
        
        // (Kh \star Khmin) (i*delta/h) = 0
        
        loss(no_h) += k1 * bin_cnt(i);
        i++;
      }
    }
    loss /= n2;
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  double compute(double h){
    int i, nb, b_1, b_2, b_3;
    double h_inv, ratio, r, r2, z, z2, k1, k2_t_2;
    double hinv_t_15, c_k1_0, c_k1_2, c_k1_4, c_k1_5, c_k1_7, c_k1_9, c_k2_0, c_k2_2, c_k2_4;
    double k3_p1, k3_p2, c_k3, c_k3_p2_0, c_k3_p2_1, c_k3_p2_2, c_k3_p2_3;
    
    nb = bin_cnt.size();
    
    double pen =  5.0 / (7.0 * h * double(n));
    
    
    
    double loss = 0;
    
    Rcpp::checkUserInterrupt();
    
    h_inv = 1 / h;
    
    
    
    b_1 = std::min(nb, int((h - hmin) / delta) + 1);
    b_2 = std::min(nb, int((h + hmin) / delta) + 1);
    b_3 = std::min(nb, int((2 * h) / delta) + 1);
    
    
    ratio = delta * h_inv;
    
    
    r = hmin * h_inv;
    r2 = r * r;
    
    
    hinv_t_15 = 15 * h_inv;
    c_k1_9 = - h_inv / 358.4;
    c_k1_7 = hinv_t_15 / 224.0;
    c_k1_5 = - hinv_t_15 / 16.0;
    c_k1_4 = hinv_t_15 / 8.0;
    c_k1_2 = - hinv_t_15 / 7.0;
    c_k1_0 = h_inv / 0.7;
    
    c_k2_0 = h_inv * (105 + r2 * (5 * r2 - 30)) / 28.0;
    c_k2_2 = h_inv * (45 * r2 - 105) / 14.0;
    c_k2_4 = 3.75 * h_inv;
    
    c_k3 = 20 * h_inv / (3584.0 * r * r2 * r2);
    
    c_k3_p2_0 = 16 * (1 + r * (r - 1) * (5 + r * (r - 4)));
    c_k3_p2_1 = -5 * (1 + r) * (5 + r * (5 * r - 14));
    c_k3_p2_2 = 3 * (1 + r * (10 + r));
    c_k3_p2_3 = 5 * (1 + r);
    
    
    
    i = 0;
    
    
    while (i <  b_1){// itu is in [0; h - hmin[
      
      
      
      
      z = i * ratio;
      
      z2 = z * z;
      
      // (Kh * Kh) (itu)
      k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
      
      // (Kh * Khmin) (itu)
      k2_t_2 = c_k2_0 + z2 * (c_k2_2 + z2 * c_k2_4);
      
      
      loss += (k1 - k2_t_2) * bin_cnt(i);
      
      
      i++;
      
      
    }
    
    while (i <  b_2){// itu is in [h - hmin; h + hmin[
      
      z = i * ratio;
      
      z2 = z * z;
      
      
      // (Kh * Kh) (itu)
      k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
      
      // (Kh * Khmin) (itu)
      k3_p1 = std::pow(1 + r - z, 5);
      k3_p2 = c_k3_p2_0 + z * (c_k3_p2_1 + z * (c_k3_p2_2 + z * (c_k3_p2_3 + z)));
      
      k2_t_2 = c_k3 * k3_p1 * k3_p2;
      
      
      loss += (k1 - k2_t_2) * bin_cnt(i);
      
      
      
      i++;
      
      
      
    }
    
    while (i <  b_3){// itu is in [h + hmin; 2 * h[
      
      z = i * ratio;
      
      z2 = z * z;
      
      
      // (Kh * Kh) (itu)
      k1 = c_k1_0 + z2 * (c_k1_2 + z2 * (c_k1_4 + z * (c_k1_5 + z2 * (c_k1_7 + z2 * c_k1_9))));
      
      // (Kh * Khmin) (itu) = 0
      
      
      loss += k1 * bin_cnt(i);
      
      i++;
      
    }
    
    
    loss /= n2;
    
    double crit = loss + pen;
    
    return(crit);
  }
  
  
};


/********************************************************************************************/
/**                           1D section doree search of optimal h                         **/
/********************************************************************************************/
 
 
// implements golden section search
double secdor_1D(criterion_1D& criterion, int nh_max, double tol){
   double hmin = criterion.hmin;
   double a = hmin;
   double b = 1.0; // hmax
   
   bool opt_find = false;
   
   // initialisation pour la recherche du min du critere
   
   double h_opt;
   int nb_h;
   if (nh_max <= 1){
     
     h_opt = 0.5 * (a + b);
     
   }
   else{
     Eigen::Vector2d crit_0;
     
     double crit_tmp;
     Eigen::Array2d H_0;
     double H;
     nb_h = 0;
     opt_find = false;
     
     double diff;
     double Lor = 0.618 * (b - a);
     H_0 << b - Lor, a + Lor;
     crit_0 = criterion.compute(H_0);
     
     
     int x_index = 0;
     
     if (crit_0(0) < crit_0(1)){
       b = H_0(1);
       H_0(1) = H_0(0);
       Lor = 0.618 * (b - a);
       H = b - Lor;
       H_0(0) = H;
       
       x_index = 1;
       crit_0(1) = crit_0(0);
     }
     else{
       a = H_0(0);
       H_0(0) = H_0(1);
       Lor = 0.618 * (b - a);
       H = a + Lor;
       
       H_0(1) = H;
       x_index = 0;
       crit_0(0) = crit_0(1);
     }
     nb_h += 2;
     
     diff = std::abs(b - a);
     
     h_opt = 0.5 * (a + b);
     opt_find = (diff < tol);
     
     while ((nb_h < nh_max)&&(!opt_find)){
       Rcpp::checkUserInterrupt();
       
       
       crit_tmp = criterion.compute(H);
       
       crit_0(1 - x_index) = crit_tmp;
       
       if (crit_0(0) < crit_0(1)){
         
         b = H_0(1);
         H_0(1) = H_0(0);
         Lor = 0.618 * (b - a);
         H = b - Lor;
         H_0(0) = H;
         
         
         x_index = 1;
         crit_0(1) = crit_0(0);
       }
       else{
         a = H_0(0);
         H_0(0) = H_0(1);
         Lor = 0.618 * (b - a);
         H = a + Lor;
         
         H_0(1) = H;
         x_index = 0;
         crit_0(0) = crit_0(1);
       }
       nb_h += 1;
       
       diff = std::abs(b - a);
       
       h_opt = 0.5 * (a + b);
       opt_find = (diff < tol);
       
     }
   }
   
   
   
   if (!opt_find){
     //warning("Warning: La recherche du minimum s'est arretee car le nombre maximum d'evaluations du critere est atteint mais la taille de l'intervalle est superieure à la tolerance");
     warning("Warning: The maximum number of evaluations has been reached but not the tolerance");
   }
   
   
   return(h_opt);
 }
 
 
// 1D golden section search
double secdor_1D_binned(binned_crit_1D& criterion, int nh_max, double tol, bool adapt_nb_bin = false) {
  double hmin = criterion.hmin;
  double a = hmin;
  double b = 1.0; // hmax
  
  
  
  
  // recherche du min
  double h_opt;
  int nb_h;
  if (nh_max <= 3){
    h_opt = 0.5 * (a + b);
    
  }
  else{
    
    
    Eigen::Array3d H_0 = Eigen::Array3d::LinSpaced(3, a, b);;
    
    Eigen::Vector3d crit_0 = criterion.compute(H_0);
    // test admissibilite
    // et ajustement du nombre de bin nb pour l'obtenir
    
    double mid_crit = 0.5 * (crit_0(0) + crit_0(2));
    if (!adapt_nb_bin){
      
      if (crit_0(1) > mid_crit){
        warning("Warning: the number of bins, nb, is probably too small. Increase nb_bin or try with adapt_nb_bin = TRUE");
      }
    }
    else{
      
      if (crit_0(1) > mid_crit){
        int nb_bin_new = criterion.nb_bin;
        
        while (crit_0(1) > mid_crit){
          Rcpp::checkUserInterrupt();
          
          nb_bin_new += criterion.xi.size() / 2;
          
          criterion.compute_tab(nb_bin_new);
          
          criterion.f_bin_cnt_diagout();
          crit_0 = criterion.compute(H_0);
          mid_crit = 0.5 * (crit_0(0) + crit_0(2));
          
        }
        Rcout << "the number of bins has been increased up to " << nb_bin_new << std::endl;
      }
      else{
        Rcout << "the number of bins has not been changed" << std::endl;
      }
      
    }
    
    
    
    Eigen::Vector2d crit_1;
    
    double crit_tmp;
    Eigen::Array2d H_1;
    double H;
    nb_h = 0;
    bool opt_find = false;
    
    double diff;
    double Lor = 0.618 * (b - a);
    H_1 << b - Lor, a + Lor;
    crit_1 = criterion.compute(H_1);
    
    
    int x_index = 0;
    
    if (crit_1(0) < crit_1(1)){
      b = H_1(1);
      H_1(1) = H_1(0);
      Lor = 0.618 * (b - a);
      H = b - Lor;
      H_1(0) = H;
      
      x_index = 1;
      crit_1(1) = crit_1(0);
    }
    else{
      a = H_1(0);
      H_1(0) = H_1(1);
      Lor = 0.618 * (b - a);
      H = a + Lor;
      
      H_1(1) = H;
      x_index = 0;
      crit_1(0) = crit_1(1);
    }
    nb_h += 2;
    
    diff = std::abs(b - a);
    
    h_opt = 0.5 * (a + b);
    opt_find = (diff < tol);
    
    while ((nb_h < nh_max)&&(!opt_find)){
      Rcpp::checkUserInterrupt();
      
      
      crit_tmp = criterion.compute(H);
      
      crit_1(1 - x_index) = crit_tmp;
      
      if (crit_1(0) < crit_1(1)){
        
        b = H_1(1);
        H_1(1) = H_1(0);
        Lor = 0.618 * (b - a);
        H = b - Lor;
        H_1(0) = H;
        
        
        x_index = 1;
        crit_1(1) = crit_1(0);
      }
      else{
        a = H_1(0);
        H_1(0) = H_1(1);
        Lor = 0.618 * (b - a);
        H = a + Lor;
        
        H_1(1) = H;
        x_index = 0;
        crit_1(0) = crit_1(1);
      }
      nb_h += 1;
      
      diff = std::abs(b - a);
      
      h_opt = 0.5 * (a + b);
      opt_find = (diff < tol);
      
    }
  }
  
  return(h_opt);
}






// [[Rcpp::export]]
double h_GK_1D_exact(Eigen::VectorXd xi, int nh_max = 40, double tol = 0.000001){
   GK_exact_crit_1D crit = GK_exact_crit_1D(xi);
   double h_opt = secdor_1D(crit, nh_max, tol);
   return(h_opt);
 }

// [[Rcpp::export]]
double h_EK_1D_exact(Eigen::VectorXd xi, int nh_max = 40, double tol = 0.000001){
  EK_exact_crit_1D crit = EK_exact_crit_1D(xi);
  double h_opt = secdor_1D(crit, nh_max, tol);
  return(h_opt);
}

// [[Rcpp::export]]
double h_BK_1D_exact(Eigen::VectorXd xi, int nh_max = 40, double tol = 0.000001){
  BK_exact_crit_1D crit = BK_exact_crit_1D(xi);
  double h_opt = secdor_1D(crit, nh_max, tol);
  return(h_opt);
}






// [[Rcpp::export]]
double h_GK_1D_bin(Eigen::VectorXd xi, int nb_bin, int nh_max = 40, double tol = 0.000001, bool adapt_nb_bin = false){
  GK_binned_crit_1D crit = GK_binned_crit_1D(xi, nb_bin);
  double h_opt = secdor_1D_binned(crit, nh_max, tol, adapt_nb_bin);
  return(h_opt);
}

// [[Rcpp::export]]
double h_EK_1D_bin(Eigen::VectorXd xi, int nb_bin, int nh_max = 40, double tol = 0.000001, bool adapt_nb_bin = false){
  EK_binned_crit_1D crit = EK_binned_crit_1D(xi, nb_bin);
  double h_opt = secdor_1D_binned(crit, nh_max, tol, adapt_nb_bin);
  return(h_opt);
}

// [[Rcpp::export]]
double h_BK_1D_bin(Eigen::VectorXd xi, int nb_bin, int nh_max = 40, double tol = 0.000001, bool adapt_nb_bin = false){
  BK_binned_crit_1D crit = BK_binned_crit_1D(xi, nb_bin);
  double h_opt = secdor_1D_binned(crit, nh_max, tol, adapt_nb_bin);
  return(h_opt);
}











/********************************************************************************************/
/**                                         multi Dim                                      **/
/********************************************************************************************/

class criterion_mD{
public:
  Eigen::MatrixXd xi;
  Eigen::MatrixXd hmin;
  Eigen::MatrixXd hmin2;
  Eigen::VectorXd hmin_diag;
  Eigen::MatrixXd P;
  Eigen::MatrixXd Pinv;
  
protected :
  int d;
  int n;
  double n2;
  
public :
  criterion_mD(Eigen::MatrixXd xi) {
    this->xi = xi;
    this->d = xi.cols();
    this->n = xi.rows();
    this->n2 = n * n;
  }
  
public:
  virtual Eigen::VectorXd compute(List H) = 0;//;
  virtual double compute(Eigen::MatrixXd H) = 0;
};


/********************************************************************************************/
/**                     multi Dim exact for Gaussian kernel                                **/
/********************************************************************************************/

class exact_crit_mD : public criterion_mD {
  
protected:
  Eigen::MatrixXd u; // u is supposed to be already squared
  int u_rows;
  
public :
  exact_crit_mD(Eigen::MatrixXd xi):criterion_mD(xi){
    this->u_rows = n * (n - 1) / 2;
    this->u = Eigen::MatrixXd::Zero(u_rows, d);
  }
  
protected :
  void outer_diff_square_mD(){
    // Computation of (oi - oj)^2 for all i and j s.t. j < i
    for (int no_d = 0; no_d < d; no_d++){
      Eigen::VectorXd xi_col = xi.col(no_d);
      int pos_u = 0;
      int n_u = n - 1;
      for (int i = 0; i < (n - 1); i++){
        Rcpp::checkUserInterrupt();
        u.block(pos_u, no_d, n_u, 1) = (xi_col.segment(i + 1, n_u).array() - xi_col(i)).square();
        pos_u += n_u;
        n_u -= 1;
      }
    }
  }
  
  
  void outer_diff_mD(){
    // Computation of (oi - oj) for all i and j s.t. j < i
    for (int no_d = 0; no_d < d; no_d++){
      Eigen::VectorXd xi_col = xi.col(no_d);
      int pos_u = 0;
      int n_u = n - 1;
      for (int i = 0; i < (n - 1); i++){
        Rcpp::checkUserInterrupt();
        u.block(pos_u, no_d, n_u, 1) = xi_col.segment(i + 1, n_u).array() - xi_col(i);
        pos_u += n_u;
        n_u -= 1;
      }
    }
  }
  
  
  
};








class GK_exact_crit_mD_diag : public exact_crit_mD{
private :
  double dlog2pi;
  double dlog2;
  double c_pen;
  Eigen::VectorXd den;
  
  
  Eigen::VectorXd K;
  double den_0;
  
  
public :
  GK_exact_crit_mD_diag(Eigen::MatrixXd xi) : exact_crit_mD(xi){
    double cst_diag = 1 / (M_SQRT2 * M_SQRT_PI * std::pow(n, 1 / double(d)));
    this->hmin = Eigen::VectorXd::Constant(d, 1, cst_diag);
    this->hmin2 = hmin.array().square();
    outer_diff_square_mD();
    this->dlog2pi = d * M_LN_2PI;
    this->dlog2 = d * M_LN2;
    this->c_pen = 1.0 / (std::pow(2.0 * M_SQRT_PI, d) * double(n));
    this->den = Eigen::VectorXd::Zero(u_rows);
    
    this->K = Eigen::VectorXd::Zero(u_rows);
  }
  
  
  
public:
  Eigen::VectorXd compute(List H){
    // dans le cas diagonal, H est une liste de vecteurs
    int nh = H.size();
    
    Eigen::VectorXd pen(nh);
    Eigen::VectorXd loss(nh);
    
    
    
    for (int no_h = 0; no_h < nh; no_h++){
      
      Eigen::VectorXd h = H(no_h);
      Eigen::VectorXd h2 = h.array().square();
      Eigen::VectorXd h2_p_hmin2 = h2 + hmin2;
      
      pen(no_h) = h.prod(); // det(h)
      
      
      den_0 = 2 * (dlog2 + dlog2pi + 2 * h.array().log().sum());
      
      // (Kh * Kh)(xi - xj)
      den = -0.25 * ((u * h2.cwiseInverse()).array() + den_0);
      den = den.array().exp();
      K = den;
      
      //(Kh * Khmin)(xi - xj)
      den_0 = h2_p_hmin2.array().log().sum() + dlog2pi;
      den = -0.5 * ((u * h2_p_hmin2.cwiseInverse()).array() + den_0);
      den = den.array().exp();
      K -= 2 * den;
      
      loss(no_h) = 2.0 * K.sum();
      
    }
    
    
    pen = pen.cwiseInverse() * c_pen;
    
    Eigen::VectorXd crit = pen + loss / n2;
    return(crit);
  }
  
  double compute(Eigen::MatrixXd H){
    // dans le cas diagonal, H est une liste de vecteurs
    
    double pen;
    double loss;
    
    Eigen::VectorXd h = H;
    Eigen::VectorXd h2 = h.array().square();
    Eigen::VectorXd h2_p_hmin2 = h2 + hmin2;
    
    pen = 1.0 / h.prod(); // 1 / det(h)
    
    
    den_0 = 2 * (dlog2 + dlog2pi + 2 * h.array().log().sum());
    
    // (Kh * Kh)(xi - xj)
    den = -0.25 * ((u * h2.cwiseInverse()).array() + den_0);
    den = den.array().exp();
    
    K = den;
    
    //(Kh * Khmin)(xi - xj)
    den_0 = h2_p_hmin2.array().log().sum() + dlog2pi;
    den = -0.5 * ((u * h2_p_hmin2.cwiseInverse()).array() + den_0);
    den = den.array().exp();
    
    K -= 2 * den;
    
    loss = 2.0 * K.sum();
    
    pen *= c_pen;
    
    double crit = pen + loss / n2;
    return(crit);
  }
  
  
};



class GK_exact_crit_mD_full : public exact_crit_mD{
  
private :
  double cst_pen;
  Eigen::VectorXd K;
  Eigen::MatrixXd L;
  Eigen::MatrixXd x;
  Eigen::VectorXd den;
  
  double d_log2pi;// = d * M_LN_2PI;
  
  
public :
  GK_exact_crit_mD_full(Eigen::MatrixXd xi, Eigen::MatrixXd S) : exact_crit_mD(xi){
    // S is the covariance matrix of xi
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(S);
    
    this->P = eigensolver.eigenvectors();
    this->Pinv = P.inverse();
    
    double cst_diag = 1 / (M_SQRT2 * M_SQRT_PI * std::pow(n, 1 / double(d)));
    Eigen::MatrixXd D = cst_diag * Eigen::MatrixXd::Identity(d, d);
    this->hmin_diag = Eigen::VectorXd::Constant(d, 1, cst_diag);
    
    this->hmin = P * D * Pinv;
    this->hmin2 = this->hmin * this->hmin;
    
    outer_diff_mD();
    
    this->cst_pen = 1 / (std::pow(2 * M_SQRT_PI, d) * double(n));
    
    this->K = Eigen::VectorXd::Zero(u_rows);
    this->L = Eigen::MatrixXd::Zero(d, d);
    this->x = Eigen::MatrixXd::Zero(d , u_rows);
    this->den = Eigen::VectorXd::Zero(u_rows);
    this->d_log2pi = d * M_LN_2PI;
    
  }
  
  
  
public:
  Eigen::VectorXd compute(List H){
    int nh = H.size();
    
    // Calcul de la penalite pour tout h de H
    Eigen::VectorXd pen = Eigen::VectorXd::Constant(nh, cst_pen);
    Eigen::VectorXd loss = Eigen::VectorXd::Zero(nh);
    
    Eigen::MatrixXd h(d, d);
    Eigen::MatrixXd h2(d,d);
    
    double sum_log_diag_L;
    for (int no_h = 0; no_h < nh; no_h++){
      Rcpp::checkUserInterrupt();
      
      h = H(no_h);
      h2 = h * h;
      
      pen(no_h) /= h.determinant();
      
      
      
      
      // Calcul de la densite gaussienne multivariée K_{sqrt(2)h}
      // Pour un calcul plus rapide on passe par la decomposition de cholesky de sig
      // Pour cela on utilise la librairie RcppEigen qui wrap la librairie c++ eigen
      
      // on recupere la matrice de la decomposition
      // L est une matrice de taille d x d
      L = (2 * h2).llt().matrixL();
      
      
      // on résoud le systeme matriciel lineaire Lx = u
      // x est une matrice de taille d x n(n-1)/2
      // u is a matrix of size n(n-1)/2 x d
      
      sum_log_diag_L = L.diagonal().array().log().sum();
      x = L.colPivHouseholderQr().solve(u.transpose());
      den = x.array().square().colwise().sum();
      den = den.array() + 2 * sum_log_diag_L + d_log2pi;
      den *= - 0.5;
      den = den.array().exp();
      K = den;
      
      
      L = (h2 + hmin2).llt().matrixL();
      
      // on résoud le systeme matriciel lineaire Lx = u
      // x est une matrice de taille d x n(n-1)/2
      // u is a matrix of size n(n-1)/2 x d
      
      sum_log_diag_L = L.diagonal().array().log().sum();
      x = L.colPivHouseholderQr().solve(u.transpose());
      den = x.array().square().colwise().sum();
      den = den.array() + 2 * sum_log_diag_L + d_log2pi;
      den *= - 0.5;
      den = den.array().exp();
      
      K -= 2 * den;
      
      
      loss(no_h) = 2 * K.sum();
      
    }
    loss /= n2;
    
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  
  
  
  
  
  double compute(Eigen::MatrixXd H){
    
    double pen = cst_pen;
    double loss = 0.0;
    
    Eigen::MatrixXd h(d, d);
    Eigen::MatrixXd h2(d,d);
    
    double sum_log_diag_L;
    
    Rcpp::checkUserInterrupt();
    
    h = H;
    h2 = h * h;
    
    
    pen /= h.determinant();
    
    // Calcul de la densite gaussienne multivariée K_{sqrt(2)h}
    // Pour un calcul plus rapide on passe par la decomposition de cholesky de sig
    // Pour cela on utilise la librairie RcppEigen qui wrap la librairie c++ eigen
    
    // on recupere la matrice de la decomposition
    // L est une matrice de taille d x d
    L = (2 * h2).llt().matrixL();
    
    // on résoud le systeme matriciel lineaire Lx = u
    // x_1 est une matrice de taille d x n(n-1)/2
    // u is a matrix of size n(n-1)/2 x d
    
    sum_log_diag_L = L.diagonal().array().log().sum();
    x = L.colPivHouseholderQr().solve(u.transpose());
    den = x.array().square().colwise().sum();
    den = den.array() + 2 * sum_log_diag_L + d_log2pi;
    den *= - 0.5;
    den = den.array().exp();
    
    K = den;
    
    L = (h2 + hmin2).llt().matrixL();
    
    sum_log_diag_L = L.diagonal().array().log().sum();
    x = L.colPivHouseholderQr().solve(u.transpose());
    den = x.array().square().colwise().sum();
    den = den.array() + 2 * sum_log_diag_L + d_log2pi;
    den *= - 0.5;
    den = den.array().exp();
    
    K -= 2 * den;
    
    loss = 2 * K.sum();
    loss /= n2;
    
    double crit = loss + pen;
    return(crit);
  }
  
  
  
};





/********************************************************************************************/
/**                           mD golden section search of optimal h                        **/
/********************************************************************************************/


Eigen::VectorXd secdor_mD_diag(criterion_mD& criterion, int nh_max, double tol) {
  
  
  
  
  Eigen::VectorXd hmin = criterion.hmin;
  int d = hmin.size();
  
  //initialisation des bornes
  Eigen::VectorXd a = hmin;
  Eigen::VectorXd b = Eigen::MatrixXd::Constant(d, 1, 1.0);
  
  // point milieu
  Eigen::VectorXd mid(d);
  mid = 0.5 * (a + b);
  
  int nb_h;
  
  
  Eigen::Vector2d crit_0;
  
  List H_0(2);
  Eigen::VectorXd H(d);
  Eigen::VectorXd h_opt;
  bool opt_find = false;
  
  if (nh_max <= 1){
    
    h_opt = mid;
    // meme si on n'a effectue aucune evaluation du critere,
    // comme on ne peut pas faire de dichotomie avec 1 seule evaluation du critere,
    // on suppose que l'optimum est le point milieu
    // nb_h = 1;
  }
  else{
    
    nb_h = 0;
    
    int current_axis = 0;
    // point milieu m dont la coordonnee sur l'axe courant est remplacee par celle du milieu - tolerance
    Eigen::VectorXd x_1(d);
    // point milieu m dont la coordonnee sur l'axe courant est remplacee par celle du milieu + tolerance
    Eigen::VectorXd x_2(d);
    
    Eigen::VectorXd Lor(d);
    Lor = 0.618 * (b - a);
    int x_index = 0;
    
    Eigen::ArrayXd diff;
    
    
    int iter_per_axe = 2;
    double crit_tmp;
    
    
    while ((nb_h < nh_max)&&(!opt_find)){
      Rcpp::checkUserInterrupt();
      current_axis = 0;
      while (current_axis < d){
        
        x_1 = mid;
        x_1(current_axis) = b(current_axis) - Lor(current_axis);
        x_2 = mid;
        x_2(current_axis) = a(current_axis) + Lor(current_axis);
        
        H_0(0) = x_1;
        H_0(1) = x_2;
        crit_0 = criterion.compute(H_0);
        
        if (crit_0(0) < crit_0(1)){
          b(current_axis) = x_2(current_axis);
          H_0(1) = H_0(0);
          Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
          H = 0.5 * (a + b);
          H(current_axis) = b(current_axis) - Lor(current_axis);
          
          H_0(0) = H;
          
          x_index = 1;
          crit_0(1) = crit_0(0);
        }
        else{
          
          a(current_axis) = x_1(current_axis);
          H_0(0) = H_0(1);
          
          Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
          
          H = 0.5 * (a + b);
          H(current_axis) = a(current_axis) + Lor(current_axis);
          
          H_0(1) = H;
          x_index = 0;
          crit_0(0) = crit_0(1);
        }
        nb_h += 2;
        
        diff = (b - a).array().abs();
        
        h_opt = 0.5 * (a + b);
        opt_find = (diff < tol).all();
        
        for (int iter = 0; iter < iter_per_axe; iter++){
          
          
          crit_tmp = criterion.compute(H);
          
          crit_0(1 - x_index) = crit_tmp;
          
          if (crit_0(0) < crit_0(1)){
            
            x_2 = H_0(1);
            b(current_axis) = x_2(current_axis);
            H_0(1) = H_0(0);
            Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
            H = 0.5 * (a + b);
            H(current_axis) = b(current_axis) - Lor(current_axis);
            
            H_0(0) = H;
            
            x_index = 1;
            crit_0(1) = crit_0(0);
          }
          else{
            x_1 = H_0(0);
            
            a(current_axis) = x_1(current_axis);
            H_0(0) = H_0(1);
            Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
            H = 0.5 * (a + b);
            H(current_axis) = a(current_axis) + Lor(current_axis);
            
            
            H_0(1) = H;
            x_index = 0;
            crit_0(0) = crit_0(1);
          }
          nb_h += 1;
          
          diff = (b - a).array().abs();
          h_opt = 0.5 * (a + b);
          opt_find = (diff < tol).all();
          
          
          
          
        }// fin de la boucle for sur les iterations par axe
        
        mid = 0.5 * (a + b);
        
        
        
        // current_axis = (current_axis + 1) % d;
        current_axis += 1;
        
      }// fin du while (current_axis < d)
    }// fin du while ((nb_h < nh_max)&&(!opt_find))
  }// fin du if-else dans le cas où nh_max > 1
  
  if (!opt_find){
    //warning("Warning: La recherche du minimum s'est arretee car le nombre maximum d'evaluations du critere est atteint mais la taille de l'intervalle est superieure à la tolerance");
    warning("Warning: The maximum number of evaluations has been reached but not the tolerance");
  }
  
  return(h_opt);
}



Eigen::MatrixXd secdor_mD_full(criterion_mD& criterion, int nh_max, double tol) {
  
  
  Eigen::MatrixXd hmin = criterion.hmin;
  int d = hmin.rows();
  
  
  //initialisation des bornes
  Eigen::VectorXd a = criterion.hmin_diag;
  Eigen::VectorXd b = Eigen::VectorXd::Constant(d, 1, 1.0);
  
  Eigen::MatrixXd P = criterion.P;
  Eigen::MatrixXd Pinv = criterion.Pinv;
  
  // point milieu
  Eigen::VectorXd mid(d);
  mid = 0.5 * (a + b);
  
  int nb_h;
  
  Eigen::Vector2d crit_0;
  
  List H_0(2);
  
  Eigen::VectorXd H_tmp(d);
  Eigen::MatrixXd H(d, d);
  Eigen::MatrixXd h_opt(d, d);
  bool opt_find = false;
  
  if (nh_max <= 1){
    
    h_opt = P * mid.asDiagonal() * Pinv;
    // meme si on n'a effectue aucune evaluation du critere,
    // comme on ne peut pas faire de dichotomie avec 1 seule evaluation du critere,
    // on suppose que l'optimum est le point milieu
    nb_h = 1;
  }
  else{
    
    nb_h = 0;
    
    int current_axis = 0;
    
    Eigen::VectorXd x_1(d);
    Eigen::MatrixXd x_1_mat(d, d);
    
    Eigen::VectorXd x_2(d);
    Eigen::MatrixXd x_2_mat(d, d);
    
    Eigen::VectorXd Lor(d);
    Lor = 0.618 * (b - a);
    int x_index = 0;
    
    Eigen::ArrayXd diff;
    
    
    int iter_per_axe = 2;
    double crit_tmp;
    
    
    while ((nb_h < nh_max)&&(!opt_find)){
      Rcpp::checkUserInterrupt();
      current_axis = 0;
      while (current_axis < d){
        
        x_1 = mid;
        x_1(current_axis) = b(current_axis) - Lor(current_axis);
        x_2 = mid;
        x_2(current_axis) = a(current_axis) + Lor(current_axis);
        
        x_1_mat = P * x_1.asDiagonal() * Pinv;
        x_2_mat = P * x_2.asDiagonal() * Pinv;
        
        H_0(0) = x_1_mat;
        H_0(1) = x_2_mat;
        
        crit_0 = criterion.compute(H_0);
        
        if (crit_0(0) < crit_0(1)){
          b(current_axis) = x_2(current_axis);
          x_2 = x_1;
          Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
          H_tmp = 0.5 * (a + b);
          H_tmp(current_axis) = b(current_axis) - Lor(current_axis);
          H = P * H_tmp.asDiagonal() * Pinv;
          x_1 = H_tmp;
          x_index = 1;
          crit_0(1) = crit_0(0);
        }
        else{
          a(current_axis) = x_1(current_axis);
          x_1 = x_2;
          Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
          H_tmp = 0.5 * (a + b);
          H_tmp(current_axis) = a(current_axis) + Lor(current_axis);
          H = P * H_tmp.asDiagonal() * Pinv;
          x_2 = H_tmp;
          x_index = 0;
          crit_0(0) = crit_0(1);
        }
        nb_h += 2;
        diff = (b - a).array().abs();
        h_opt = P * (0.5 * (a + b)).asDiagonal() * Pinv;
        opt_find = (diff < tol).all();
        
        for (int iter = 0; iter < iter_per_axe; iter++){
          crit_tmp = criterion.compute(H);
          crit_0(1 - x_index) = crit_tmp;
          if (crit_0(0) < crit_0(1)){
            b(current_axis) = x_2(current_axis);
            x_2 = x_1;
            Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
            H_tmp = 0.5 * (a + b);
            H_tmp(current_axis) = b(current_axis) - Lor(current_axis);
            H = P * H_tmp.asDiagonal() * Pinv;
            x_1 = H_tmp;
            x_index = 1;
            crit_0(1) = crit_0(0);
          }
          else{
            a(current_axis) = x_1(current_axis);
            x_1 = x_2;
            Lor(current_axis) = 0.618 * (b(current_axis) - a(current_axis));
            H_tmp = 0.5 * (a + b);
            H_tmp(current_axis) = a(current_axis) + Lor(current_axis);
            H = P * H_tmp.asDiagonal() * Pinv;
            x_2 = H_tmp;
            x_index = 0;
            crit_0(0) = crit_0(1);
          }
          nb_h += 1;
          diff = (b - a).array().abs();
          h_opt = P * (0.5 * (a + b)).asDiagonal() * Pinv;
          opt_find = (diff < tol).all();
          
        }// fin de la boucle for sur les iterations par axe
        
        mid = 0.5 * (a + b);
        // current_axis = (current_axis + 1) % d;
        current_axis += 1;
        
      }// fin du while (current_axis < d)
    }// fin du while ((nb_h < nh_max)&&(!opt_find))
  }// fin du if-else dans le cas où nh_max > 1
  
  
  if (!opt_find){
    //warning("Warning: La recherche du minimum s'est arretee car le nombre maximum d'evaluations du critere est atteint mais la taille de l'intervalle est superieure à la tolerance");
    warning("Warning: The maximum number of evaluations has been reached but not the tolerance");
  }
  return(h_opt);
}







// [[Rcpp::export]]
Eigen::VectorXd h_GK_mD_diag_exact(Eigen::MatrixXd x_i, int nh_max, double tol){
  GK_exact_crit_mD_diag crit = GK_exact_crit_mD_diag(x_i);
  Eigen::VectorXd h_opt = secdor_mD_diag(crit, nh_max, tol);
  return(h_opt);
}




// [[Rcpp::export]]
Eigen::MatrixXd h_GK_mD_full_exact(Eigen::MatrixXd x_i, Eigen::MatrixXd S, int nh_max, double tol){
  GK_exact_crit_mD_full crit = GK_exact_crit_mD_full(x_i, S);
  Eigen::MatrixXd h_opt = secdor_mD_full(crit, nh_max, tol);
  return(h_opt);
}


/********************************************************************************************/
/**                     multi Dim binned for Gaussian kernel                                **/
/********************************************************************************************/


class binned_crit_mD : public criterion_mD {
  
  
protected:
  int size_bin_weights;
  Eigen::VectorXd nb_bin_vect;// pour un nombre de bin different par axe
  Eigen::MatrixXd xi;
  Eigen::VectorXd delta;
  Eigen::VectorXd max_grid;
  Eigen::VectorXd min_grid;
  
  Eigen::VectorXd bin_w_inpairs_prod;
  
  Eigen::MatrixXd grid;// = (bin_points * delta).square() for diag case and (bin_points * delta) for full case
  
  Eigen::ArrayXd nb_bin_head_prod;
  
  
  
  
public :
  binned_crit_mD(Eigen::MatrixXd xi, int nb_bin_per_axis, Rcpp::Nullable<Eigen::VectorXd> nb_bin_vect_, int type):criterion_mD(xi){
    
    
    if (nb_bin_vect_.isNotNull()) {
      this->nb_bin_vect = Rcpp::as<Eigen::VectorXd>(nb_bin_vect_);
    }else{
      this->nb_bin_vect = Eigen::VectorXd::Constant(d, nb_bin_per_axis);
    }
    
    this->xi = xi;
    
    
    this->max_grid = xi.colwise().maxCoeff();// * 1.01;
    this->min_grid = xi.colwise().minCoeff();// * 1.01;
    this->delta = 1.01 * (max_grid - min_grid).array() * (this->nb_bin_vect - Eigen::VectorXd::Ones(d)).array().inverse();
    
    this->size_bin_weights = this->nb_bin_vect.prod();//std::pow(nb_bin, d);
    
    bin_w_inpairs_prod = Eigen::VectorXd::Zero(size_bin_weights);
    
    
    
    f_linbin_cnt_mD_4(type);
    
  }
protected:
  
  
  
  
  // linear binning avec nombre de bin different sur chaque axe
  // on calcule le poids de chaque bin
  // diagonal terms are included
  // tentative d'acceleration en supprimant des var intermediaires et en calculant coeff avant la boucle sur n
  // le calcul de bin_points et grid se fait par des replication au lieu d'un calcul
  void f_linbin_cnt_mD_4(int type){
    
    
    // min_grid = min(xi)
    // max_grid = max(xi)
    // calcul des poids de chaque bin
    Eigen::MatrixXd m_z(n, d);
    m_z = xi.rowwise() + (0.005 * (max_grid - min_grid) - min_grid).transpose();
    m_z = m_z.array().rowwise() * delta.array().inverse().transpose();
    
    // quotient de la division entiere de xi-min par delta : xi-min = q * delta + r
    // m_q = q
    Eigen::MatrixXd m_q(n, d);
    m_q = m_z.array().floor();
    
    // reste de la division entiere de xi-min par delta divise par delta : (xi-min)/delta = q + r / delta
    // m_r = r / delta
    Eigen::MatrixXd m_r(n, d);
    m_r = m_z - m_q;
    
    
    // on construit un plan factoriel complet de d variables à 2 niveaux (0 et 1)
    // nb_nodes correspond au nombre de sommets de la grille de binning auxquels
    // il faut ajouter la contribution d'un point xi
    // cad le nombre de sommets d'un hypercube de dimension d
    int nb_nodes = std::pow(2, d);
    Eigen::MatrixXd plan(nb_nodes, d);
    for (int no_d=0; no_d<d; no_d++){
      int size = std::pow(2, no_d + 1);
      Eigen::VectorXd vect = Eigen::VectorXd::Zero(size);
      vect.tail(size/2) = Eigen::VectorXd::Ones(size/2);
      plan.col(no_d) = vect.replicate(d - no_d, 1);
    }
    
    
    
    
    int size_bin_weights = nb_bin_vect.prod();//std::pow(nb_bin, d);
    Eigen::VectorXd bin_weights = Eigen::VectorXd::Zero(size_bin_weights);
    Eigen::ArrayXXd coeff = plan.colwise().reverse().array();
    Eigen::ArrayXd cotes(d);
    
    
    this->nb_bin_head_prod = Eigen::ArrayXd::Zero(d);
    
    for (int no_d = 0; no_d < d; no_d++){
      this->nb_bin_head_prod(no_d) = nb_bin_vect.head(no_d).prod();
    }
    
    int index_w;
    for (int i=0; i<n; i++){
      for (int no_node = 0; no_node < nb_nodes; no_node++){
        index_w = ((m_q.row(i) + plan.row(no_node)).transpose().array() * nb_bin_head_prod).sum();
        cotes = coeff.row(no_node) + Eigen::pow(-1, coeff.row(no_node)) * m_r.row(i).array();
        bin_weights(index_w) += cotes.prod();
      }
    }
    
    
    
    // sauvegarde des points du maillage dans grid
    //et bin_points (bin_points n'est pas sauvegarde)
    // grid correspond a bin_points * delta
    Eigen::MatrixXd bin_points = Eigen::MatrixXd::Zero(nb_bin_vect.prod(), d);
    
    int pos = 0;
    for (int no_d = 0; no_d < d; no_d++){
      pos = 0;
      int v_size = nb_bin_head_prod(no_d);
      for (int no_bin = 0; no_bin < nb_bin_vect(no_d); no_bin++){
        
        bin_points.col(no_d).segment(pos, v_size) = Eigen::ArrayXd::Constant(v_size, no_bin);
        pos += v_size;
      }
      int nb_rep = nb_bin_vect.tail(d - no_d - 1).prod() - 1;//std::pow(nb_bin, d - 1 - no_d);
      bin_points.col(no_d).tail(bin_points.rows() - pos) = bin_points.col(no_d).head(pos).replicate(nb_rep - 1, 1);
    }
    
    if (type == 0){// cas h diagonal : grid est mis au carre
      grid = (bin_points.array().rowwise() * delta.transpose().array()).square().matrix();
    }
    else{// cas h pleine
      grid = (bin_points.array().rowwise() * delta.transpose().array()).matrix();
    }
    
    
    
    // calcul du produit des poids N_k * N_l  assigne a bin_weights2(k-l) et diagonal terms included
    Eigen::ArrayXd i(d);
    Eigen::ArrayXd j(d);
    double ci;
    int index_diff_ij;
    for (int ii = 0; ii < bin_weights.size(); ii++){
      Rcpp::checkUserInterrupt();
      ci = bin_weights(ii);
      bin_w_inpairs_prod(0) += ci * ci;
      i = bin_points.row(ii);
      for (int jj = 0; jj < ii; jj++){
        j = bin_points.row(jj);
        index_diff_ij = ((i - j).abs() * nb_bin_head_prod).sum();
        bin_w_inpairs_prod(index_diff_ij) += ci * bin_weights(jj);
      }
    }
    bin_w_inpairs_prod(0) *= 0.5;
    
  }
  
  
  
  
  
  
  
};





class GK_binned_crit_mD_diag : public binned_crit_mD{
private :
  double dlog2pi;
  double dlog2;
  double c_pen;
  
  
  
  
public :
  GK_binned_crit_mD_diag(Eigen::MatrixXd xi,  Rcpp::Nullable<Eigen::VectorXd> nb_bin_vect_, int nb_bin_per_axis=32) : binned_crit_mD(xi, nb_bin_per_axis, nb_bin_vect_, 0){
    double cst_diag = 1 / (M_SQRT2 * M_SQRT_PI * std::pow(n, 1 / double(d)));
    this->hmin = Eigen::VectorXd::Constant(d, 1, cst_diag);
    this->hmin2 = hmin.array().square();
    this->dlog2pi = d * M_LN_2PI;
    this->dlog2 = d * M_LN2;
    this->c_pen = std::pow(M_1_SQRT_2PI, d) * 2.0 / double(n);
    
    
  }
  
  
  
public:
  Eigen::VectorXd compute(List H){
    
    int nh = H.size();
    
    Eigen::VectorXd pen(nh);
    Eigen::VectorXd loss(nh);
    
    Eigen::VectorXd h(d);
    Eigen::VectorXd h2(d);
    Eigen::VectorXd D(d);
    
    Eigen::VectorXd L(d);
    
    Eigen::ArrayX2d L1_tmp(d, 2);
    Eigen::ArrayX2d L2_tmp(d, 2);
    L1_tmp.col(0) = nb_bin_vect.array() - 1;
    L2_tmp.col(0) = nb_bin_vect.array() - 1;
    Eigen::ArrayX2d L_tmp(d, 2);
    
    double den_0, tau_lambda1_sqrt, tau_lambda2_sqrt;
    double den_0_K1 = 0.5 * (dlog2 + dlog2pi);
    
    // pour le calcul des indices de grid et bin_w_inpairs_prod a conserver
    int nb_bloc, v_size, nb_rep, pos, bloc_size, nb_ex_lines;
    Eigen::ArrayXi bloc_index;//, v;
    
    for (int no_h = 0; no_h < nh; no_h++){
      
      Rcpp::checkUserInterrupt();
      h = H(no_h);
      h2 = h.array().square();
      D = h2 + hmin2;
      
      pen(no_h) = D.prod();//determinant of D
      
      // H etant diagonale, ses valeurs propres sont égales à ses éléments diagonaux.
      
      // pour Kh*Kh
      tau_lambda1_sqrt = 3.7 * std::sqrt(M_SQRT2 * h.maxCoeff());
      // pour Kh*Khmin
      tau_lambda2_sqrt = 3.7 * std::sqrt(std::sqrt(D.maxCoeff()));
      
      L1_tmp.col(1) = (tau_lambda1_sqrt * delta.cwiseInverse()).array().ceil();
      L2_tmp.col(1) = (tau_lambda2_sqrt * delta.cwiseInverse()).array().ceil();
      
      L_tmp.col(0) = L1_tmp.rowwise().minCoeff();
      L_tmp.col(1) = L2_tmp.rowwise().minCoeff();
      
      L = L_tmp.rowwise().maxCoeff().matrix();
      
      
      if ((nb_bin_vect.array() - L.array() - 1 > 0).any()){
        
        nb_bloc = (L.tail(d - 1).array() + 1).prod();
        bloc_size = L(0) + 1;
        nb_ex_lines = bloc_size * nb_bloc;
        pos = 0;
        
        RowMajorMatrixXd extracted_grid(nb_ex_lines, d);
        
        Eigen::VectorXd extracted_bin_w(nb_ex_lines);
        Eigen::VectorXd K(nb_ex_lines);
        Eigen::VectorXd den(nb_ex_lines);
        
        bloc_index = Eigen::ArrayXi::Zero(nb_bloc);
        
        for (int no_d = 1; no_d < d; no_d++){
          pos = 0;
          v_size = (L.tail(d - 1).array() + 1).head(no_d - 1).prod();
          for (int no_bin = 0; no_bin <= L(no_d); no_bin++){
            bloc_index.segment(pos, v_size) += Eigen::ArrayXi::Constant(v_size, no_bin) * int(nb_bin_head_prod(no_d));
            pos += v_size;
          }
          nb_rep = (L.array() + 1).tail(d - no_d - 1).prod() - 1;
          bloc_index.tail(nb_bloc - pos) = bloc_index.head(pos).replicate(nb_rep - 1, 1);
        }
        
        // dans la prochaine version de eigen il devrait etre possible d'extraire une sous matrice
        // à partir d'un tableau d'indices
        pos = 0;
        for (int no_bloc = 0; no_bloc < nb_bloc; no_bloc++){
          extracted_grid.block(pos, 0, bloc_size, d) = grid.block(bloc_index(no_bloc), 0, bloc_size, d);
          extracted_bin_w.segment(pos, bloc_size) = bin_w_inpairs_prod.segment(bloc_index(no_bloc), bloc_size);
          pos += bloc_size;
        }
        
        
        // (Kh * Kh)(xi - xj)
        den_0 = den_0_K1 + h.array().log().sum();
        // grid is already squared
        den = - ((extracted_grid * 0.25 * h2.cwiseInverse()).array() + den_0);
        den = den.array().exp();
        K = den;
        
        // (Kh * Khmin)(xi - xj)
        den_0 = D.array().log().sum() + dlog2pi;
        // grid is already squared
        den = - 0.5 * ((extracted_grid * D.cwiseInverse()).array() + den_0);
        den = den.array().exp();
        K -= 2 * den;
        
        // on multiplie par 2 car on ne calcule que les produits NkNl pour k<l
        loss(no_h) = 2 * K.dot(extracted_bin_w);
        
      }
      else{
        
        Eigen::VectorXd K(int(nb_bin_vect.prod()));
        Eigen::VectorXd den(int(nb_bin_vect.prod()));
        
        // (Kh * Kh)(xi - xj)
        den_0 = den_0_K1 + h.array().log().sum();
        // grid is already squared
        den = - ((grid * 0.25 * h2.cwiseInverse()).array() + den_0);
        den = den.array().exp();
        K = den;
        
        // (Kh * Khmin)(xi - xj)
        den_0 = D.array().log().sum() + dlog2pi;
        // grid is already squared
        den = - 0.5 * ((grid * D.cwiseInverse()).array() + den_0);
        den = den.array().exp();
        K -= 2 * den;
        
        // on multiplie par 2 car on ne calcule que les produits NkNl pour k<l
        loss(no_h) = 2 * K.dot(bin_w_inpairs_prod);// le *2 car on a calcule NkNl pour k=0,...,nb et l<k
        
      }
    }
    
    pen = pen.cwiseSqrt();
    pen = pen.cwiseInverse() * c_pen;
    
    Eigen::VectorXd crit = pen + loss / n2;
    return(crit);
    
  }
  
  
  double compute(Eigen::MatrixXd H){
    
    double pen;
    double loss;
    
    Eigen::VectorXd h(d);
    Eigen::VectorXd h2(d);
    Eigen::VectorXd D(d);
    
    Eigen::VectorXd L(d);
    
    Eigen::ArrayX2d L1_tmp(d, 2);
    Eigen::ArrayX2d L2_tmp(d, 2);
    L1_tmp.col(0) = nb_bin_vect.array() - 1;
    L2_tmp.col(0) = nb_bin_vect.array() - 1;
    Eigen::ArrayX2d L_tmp(d, 2);
    
    
    double den_0, tau_lambda1_sqrt, tau_lambda2_sqrt;
    
    
    double den_0_K1 = 0.5 * (dlog2 + dlog2pi);
    
    // pour le calcul des indices de grid et bin_w_inpairs_prod a conserver
    int nb_bloc, v_size, nb_rep, pos, bloc_size, nb_ex_lines;
    Eigen::ArrayXi bloc_index;
    
    Rcpp::checkUserInterrupt();
    h = H;
    h2 = h.array().square();
    D = h2 + hmin2;
    
    
    pen = c_pen / std::sqrt(D.prod());// = c_pen / sqrt(determinant of D)
    
    
    // H etant diagonale, ses valeurs propres sont égales à ses éléments diagonaux.
    
    // pour Kh*Kh
    tau_lambda1_sqrt = 3.7 * std::sqrt(M_SQRT2 * h.maxCoeff());
    // pour Kh*Khmin
    tau_lambda2_sqrt = 3.7 * std::sqrt(std::sqrt(D.maxCoeff()));
    
    L1_tmp.col(1) = (tau_lambda1_sqrt * delta.cwiseInverse()).array().ceil();
    L2_tmp.col(1) = (tau_lambda2_sqrt * delta.cwiseInverse()).array().ceil();
    
    L_tmp.col(0) = L1_tmp.rowwise().minCoeff();
    L_tmp.col(1) = L2_tmp.rowwise().minCoeff();
    
    L = L_tmp.rowwise().maxCoeff().matrix();
    
    if ((nb_bin_vect.array() - L.array() - 1 > 0).any()){
      
      nb_bloc = (L.tail(d - 1).array() + 1).prod();
      bloc_size = L(0) + 1;
      nb_ex_lines = bloc_size * nb_bloc;
      pos = 0;
      
      RowMajorMatrixXd extracted_grid(nb_ex_lines, d);
      
      Eigen::VectorXd extracted_bin_w(nb_ex_lines);
      Eigen::VectorXd K(nb_ex_lines);
      Eigen::VectorXd den(nb_ex_lines);
      
      bloc_index = Eigen::ArrayXi::Zero(nb_bloc);
      
      for (int no_d = 1; no_d < d; no_d++){
        pos = 0;
        v_size = (L.tail(d - 1).array() + 1).head(no_d - 1).prod();
        
        for (int no_bin = 0; no_bin <= L(no_d); no_bin++){
          bloc_index.segment(pos, v_size) += Eigen::ArrayXi::Constant(v_size, no_bin) * int(nb_bin_head_prod(no_d));
          pos += v_size;
        }
        nb_rep = (L.array() + 1).tail(d - no_d - 1).prod() - 1;
        bloc_index.tail(nb_bloc - pos) = bloc_index.head(pos).replicate(nb_rep - 1, 1);
      }
      
      // dans la prochaine version de eigen il devrait etre possible d'extraire une sous matrice
      // à partir d'un tableau d'indices
      pos = 0;
      for (int no_bloc = 0; no_bloc < nb_bloc; no_bloc++){
        extracted_grid.block(pos, 0, bloc_size, d) = grid.block(bloc_index(no_bloc), 0, bloc_size, d);
        extracted_bin_w.segment(pos, bloc_size) = bin_w_inpairs_prod.segment(bloc_index(no_bloc), bloc_size);
        pos += bloc_size;
      }
      
      
      
      // (Kh * Kh)(xi - xj)
      den_0 = den_0_K1 + h.array().log().sum();
      // grid is already squared
      den = - ((extracted_grid * 0.25 * h2.cwiseInverse()).array() + den_0);
      den = den.array().exp();
      K = den;
      
      // (Kh * Khmin)(xi - xj)
      den_0 = D.array().log().sum() + dlog2pi;
      // grid is already squared
      den = - 0.5 * ((extracted_grid * D.cwiseInverse()).array() + den_0);
      den = den.array().exp();
      K -= 2 * den;
      
      // on multiplie par 2 car on ne calcule que les produits NkNl pour k<l
      loss = 2 * K.dot(extracted_bin_w);
      
    }
    else{
      
      Eigen::VectorXd K(int(nb_bin_vect.prod()));
      Eigen::VectorXd den(int(nb_bin_vect.prod()));
      
      // (Kh * Kh)(xi - xj)
      den_0 = den_0_K1 + h.array().log().sum();
      // grid is already squared
      den = - ((grid * 0.25 * h2.cwiseInverse()).array() + den_0);
      den = den.array().exp();
      K = den;
      
      
      // (Kh * Khmin)(xi - xj)
      den_0 = D.array().log().sum() + dlog2pi;
      // grid is already squared
      den = - 0.5 * ((grid * D.cwiseInverse()).array() + den_0);
      den = den.array().exp();
      K -= 2 * den;
      
      
      // on multiplie par 2 car on ne calcule que les produits NkNl pour k<l
      loss = 2 * K.dot(bin_w_inpairs_prod);// le *2 car on a calcule NkNl pour k=0,...,nb et l<k
      
    }
    
    
    double crit = pen + loss / n2;
    return(crit);
  }
  
};





class GK_binned_crit_mD_full : public binned_crit_mD{
private:
  Eigen::VectorXd K;
  Eigen::MatrixXd L;
  Eigen::MatrixXd x;
  Eigen::VectorXd den;
  double d_log2pi;
  double cst_pen;
  
public :
  GK_binned_crit_mD_full(Eigen::MatrixXd xi, Eigen::MatrixXd S, int nb_bin_per_axis, Rcpp::Nullable<Eigen::VectorXd> nb_bin_2) : binned_crit_mD(xi, nb_bin_per_axis, nb_bin_2, 1){
    // S is the covariance matrix of xi
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(S);
    this->P = eigensolver.eigenvectors();
    this->Pinv = P.inverse();
    
    double cst_diag = 1 / (M_SQRT2 * M_SQRT_PI * std::pow(n, 1 / double(d)));
    Eigen::MatrixXd D = cst_diag * Eigen::MatrixXd::Identity(d, d);
    this->hmin_diag = Eigen::VectorXd::Constant(d, 1, cst_diag);
    this->hmin = P * D * Pinv;
    this->hmin2 = this->hmin * this->hmin;
    
    
    this->cst_pen = 2.0 * std::pow(M_1_SQRT_2PI, d) / double(n);
    
    
    this->K = Eigen::VectorXd::Zero(size_bin_weights);
    
    this->x = Eigen::MatrixXd::Zero(d , size_bin_weights);
    this->den = Eigen::VectorXd::Zero(size_bin_weights);
    this->d_log2pi = d * M_LN_2PI;
  }
  
  
  
public:
  Eigen::VectorXd compute(List H){
    
    int nh = H.size();
    Eigen::VectorXd pen = Eigen::VectorXd::Constant(nh, cst_pen);
    Eigen::VectorXd loss(nh);
    
    Eigen::MatrixXd h(d, d);
    Eigen::MatrixXd h2(d, d);
    
    Eigen::MatrixXd L(d, d);
    
    Eigen::VectorXd M(d);
    
    Eigen::ArrayX2d M1_tmp(d, 2);
    Eigen::ArrayX2d M2_tmp(d, 2);
    M1_tmp.col(0) = nb_bin_vect.array() - 1;
    M2_tmp.col(0) = nb_bin_vect.array() - 1;
    Eigen::ArrayX2d M_tmp(d, 2);
    
    double sum_log_diag_L, tau_lambda1_sqrt, tau_lambda2_sqrt;
    
    // pour le calcul des indices de grid et bin_w_inpairs_prod a conserver
    int nb_bloc, v_size, nb_rep, pos, bloc_size, nb_ex_lines;
    Eigen::ArrayXi bloc_index;
    
    for (int no_h = 0; no_h < nh; no_h++){
      Rcpp::checkUserInterrupt();
      
      h = H(no_h);
      h2 = h * h;
      
      
      
      
      // pour Kh*Kh
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_Sig2_1(2 * h2);
      tau_lambda1_sqrt = 3.7 * std::sqrt(eigensolver_Sig2_1.eigenvalues().maxCoeff());
      // pour Kh*Khmin
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_Sig2_2(h2 + hmin2);
      tau_lambda2_sqrt = 3.7 * std::sqrt(eigensolver_Sig2_2.eigenvalues().maxCoeff());
      
      
      
      M1_tmp.col(1) = (tau_lambda1_sqrt * delta.cwiseInverse()).array().ceil();
      M2_tmp.col(1) = (tau_lambda2_sqrt * delta.cwiseInverse()).array().ceil();
      
      M_tmp.col(0) = M1_tmp.rowwise().minCoeff();
      M_tmp.col(1) = M2_tmp.rowwise().minCoeff();
      
      M = M_tmp.rowwise().maxCoeff().matrix();
      
      
      
      // on verifie que il y a bien au moins 1 axe sur lequel on peut reduire le nombre de calculs
      // if there exist a pair i,j such that nb_bin_vect(i,j) - M(i,j) - 1  > 0
      if ((nb_bin_vect.array() - M.array() - 1 > 0).any()){
        
        nb_bloc = (M.tail(d - 1).array() + 1).prod();
        bloc_size = M(0) + 1;
        nb_ex_lines = bloc_size * nb_bloc;
        pos = 0;
        
        RowMajorMatrixXd extracted_grid(nb_ex_lines, d);
        Eigen::VectorXd extracted_bin_w(nb_ex_lines);
        
        bloc_index = Eigen::ArrayXi::Zero(nb_bloc);
        
        for (int no_d = 1; no_d < d; no_d++){
          pos = 0;
          v_size = (M.tail(d - 1).array() + 1).head(no_d - 1).prod();
          for (int no_bin = 0; no_bin <= M(no_d); no_bin++){
            bloc_index.segment(pos, v_size) += Eigen::ArrayXi::Constant(v_size, no_bin) * int(nb_bin_head_prod(no_d));
            pos += v_size;
          }
          nb_rep = (M.array() + 1).tail(d - no_d - 1).prod() - 1;
          bloc_index.tail(nb_bloc - pos) = bloc_index.head(pos).replicate(nb_rep - 1, 1);
        }
        
        // dans la prochaine version de eigen il devrait etre possible d'extraire une sous matrice
        // à partir d'un tableau d'indices
        pos = 0;
        for (int no_bloc = 0; no_bloc < nb_bloc; no_bloc++){
          extracted_grid.block(pos, 0, bloc_size, d) = grid.block(bloc_index(no_bloc), 0, bloc_size, d);
          extracted_bin_w.segment(pos, bloc_size) = bin_w_inpairs_prod.segment(bloc_index(no_bloc), bloc_size);
          pos += bloc_size;
        }
        
        
        // Calcul de la densite gaussienne multivariée K_{sqrt(2)h}
        // Pour un calcul plus rapide on passe par la decomposition de cholesky de 2*h2
        // Pour cela on utilise la librairie RcppEigen qui wrap la librairie c++ eigen
        
        // on recupere la matrice de la decomposition
        L = (2 * h2).llt().matrixL();
        
        // det(2*h2) = det(L)^2, donc log(det(2*h2)) = 2log(det(L))
        // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
        // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
        sum_log_diag_L = L.diagonal().array().log().sum();
        // on résoud le systeme matriciel lineaire Lx = u
        x = L.colPivHouseholderQr().solve(extracted_grid.transpose());
        
        // den est un vecteur de taille n(n-1)/2
        den = x.array().square().colwise().sum();
        den = den.array() + 2 * sum_log_diag_L + d_log2pi;
        den *= - 0.5;
        den = den.array().exp();
        
        K = den;
        
        
        L = (h2 + hmin2).llt().matrixL();
        
        // det(h2 + hmin2) = det(L)^2,
        // donc sqrt(det(h2 + hmin2) = det(L)
        // donc log(sqrt(det(h2 + hmin2))) = 2log(det(L))/2 = log(det(L))
        // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
        // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
        sum_log_diag_L = L.diagonal().array().log().sum();
        
        pen(no_h) /= std::exp(sum_log_diag_L);
        
        // on résoud le systeme matriciel lineaire Lx = u
        x = L.colPivHouseholderQr().solve(extracted_grid.transpose());
        
        // den est un vecteur de taille n(n-1)/2
        den = x.array().square().colwise().sum();
        den = den.array() + 2 * sum_log_diag_L + d_log2pi;
        den *= - 0.5;
        den = den.array().exp();
        
        K -= 2 * den;
        
        loss(no_h) = 2 * K.dot(extracted_bin_w);
        
      }
      else{
        
        // Calcul de la densite gaussienne multivariée K_{sqrt(2)h}
        // Pour un calcul plus rapide on passe par la decomposition de cholesky de 2*h2
        // Pour cela on utilise la librairie RcppEigen qui wrap la librairie c++ eigen
        
        // on recupere la matrice de la decomposition
        L = (2 * h2).llt().matrixL();
        
        
        // det(2*h2) = det(L)^2, donc log(det(2*h2)) = 2log(det(L))
        // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
        // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
        sum_log_diag_L = L.diagonal().array().log().sum();
        
        
        // on résoud le systeme matriciel lineaire Lx = u
        x = L.colPivHouseholderQr().solve(this->grid.transpose());
        // den est un vecteur de taille n(n-1)/2
        den = x.array().square().colwise().sum();
        den = den.array() + 2 * sum_log_diag_L + d_log2pi;
        den *= - 0.5;
        den = den.array().exp();
        
        K = den;
        
        
        L = (h2 + hmin2).llt().matrixL();
        
        // det(h2 + hmin2) = det(L)^2,
        // donc sqrt(det(h2 + hmin2) = det(L)
        // donc log(sqrt(det(h2 + hmin2))) = 2log(det(L))/2 = log(det(L))
        // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
        // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
        sum_log_diag_L = L.diagonal().array().log().sum();
        
        pen(no_h) /= std::exp(sum_log_diag_L);
        
        // on résoud le systeme matriciel lineaire Lx = u
        x = L.colPivHouseholderQr().solve(this->grid.transpose());
        // den est un vecteur de taille n(n-1)/2
        den = x.array().square().colwise().sum();
        den = den.array() + 2 * sum_log_diag_L + d_log2pi;
        den *= - 0.5;
        den = den.array().exp();
        
        K -= 2 * den;
        
        loss(no_h) = 2 * K.dot(this->bin_w_inpairs_prod);
        
      }
    }
    loss /= n2;
    Eigen::VectorXd crit = loss + pen;
    return(crit);
  }
  
  double compute(Eigen::MatrixXd H){
    
    double pen = cst_pen;
    double loss;
    
    Eigen::MatrixXd h(d, d);
    Eigen::MatrixXd h2(d, d);
    
    Eigen::MatrixXd L(d, d);
    
    Eigen::VectorXd M(d);
    
    Eigen::ArrayX2d M1_tmp(d, 2);
    Eigen::ArrayX2d M2_tmp(d, 2);
    M1_tmp.col(0) = nb_bin_vect.array() - 1;
    M2_tmp.col(0) = nb_bin_vect.array() - 1;
    Eigen::ArrayX2d M_tmp(d, 2);
    
    double sum_log_diag_L, tau_lambda1_sqrt, tau_lambda2_sqrt;
    
    // pour le calcul des indices de grid et bin_w_inpairs_prod a conserver
    int nb_bloc, v_size, nb_rep, pos, bloc_size, nb_ex_lines;
    Eigen::ArrayXi bloc_index;
    
    Rcpp::checkUserInterrupt();
    
    h = H;
    h2 = h * h;
    
    // pour Kh*Kh
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_Sig2_1(2 * h2);
    tau_lambda1_sqrt = 3.7 * std::sqrt(eigensolver_Sig2_1.eigenvalues().maxCoeff());
    // pour Kh*Khmin
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver_Sig2_2(h2 + hmin2);
    tau_lambda2_sqrt = 3.7 * std::sqrt(eigensolver_Sig2_2.eigenvalues().maxCoeff());
    
    M1_tmp.col(1) = (tau_lambda1_sqrt * delta.cwiseInverse()).array().ceil();
    M2_tmp.col(1) = (tau_lambda2_sqrt * delta.cwiseInverse()).array().ceil();
    
    M_tmp.col(0) = M1_tmp.rowwise().minCoeff();
    M_tmp.col(1) = M2_tmp.rowwise().minCoeff();
    
    M = M_tmp.rowwise().maxCoeff().matrix();
    
    
    
    // on verifie que il y a bien au moins 1 axe sur lequel on peut reduire le nombre de calculs
    // if there exist a pair i,j such that nb_bin_vect(i,j) - M(i,j) - 1  > 0
    if ((nb_bin_vect.array() - M.array() - 1 > 0).any()){
      
      nb_bloc = (M.tail(d - 1).array() + 1).prod();
      bloc_size = M(0) + 1;
      nb_ex_lines = bloc_size * nb_bloc;
      pos = 0;
      
      RowMajorMatrixXd extracted_grid(nb_ex_lines, d);
      Eigen::VectorXd extracted_bin_w(nb_ex_lines);
      
      bloc_index = Eigen::ArrayXi::Zero(nb_bloc);
      
      for (int no_d = 1; no_d < d; no_d++){
        pos = 0;
        v_size = (M.tail(d - 1).array() + 1).head(no_d - 1).prod();
        for (int no_bin = 0; no_bin <= M(no_d); no_bin++){
          bloc_index.segment(pos, v_size) += Eigen::ArrayXi::Constant(v_size, no_bin) * int(nb_bin_head_prod(no_d));
          pos += v_size;
        }
        nb_rep = (M.array() + 1).tail(d - no_d - 1).prod() - 1;
        bloc_index.tail(nb_bloc - pos) = bloc_index.head(pos).replicate(nb_rep - 1, 1);
      }
      
      // dans la prochaine version de eigen il devrait etre possible d'extraire une sous matrice
      // à partir d'un tableau d'indices
      pos = 0;
      for (int no_bloc = 0; no_bloc < nb_bloc; no_bloc++){
        extracted_grid.block(pos, 0, bloc_size, d) = grid.block(bloc_index(no_bloc), 0, bloc_size, d);
        extracted_bin_w.segment(pos, bloc_size) = bin_w_inpairs_prod.segment(bloc_index(no_bloc), bloc_size);
        pos += bloc_size;
      }
      
      
      // Calcul de la densite gaussienne multivariée K_{sqrt(2)h}
      // Pour un calcul plus rapide on passe par la decomposition de cholesky de 2*h2
      // Pour cela on utilise la librairie RcppEigen qui wrap la librairie c++ eigen
      
      // on recupere la matrice de la decomposition
      L = (2 * h2).llt().matrixL();
      
      // det(2*h2) = det(L)^2, donc log(det(2*h2)) = 2log(det(L))
      // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
      // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
      sum_log_diag_L = L.diagonal().array().log().sum();
      
      // on résoud le systeme matriciel lineaire Lx = u
      x = L.colPivHouseholderQr().solve(extracted_grid.transpose());
      // den est un vecteur de taille n(n-1)/2
      den = x.array().square().colwise().sum();
      den = den.array() + 2 * sum_log_diag_L + d_log2pi;
      den *= - 0.5;
      den = den.array().exp();
      
      K = den;
      
      
      L = (h2 + hmin2).llt().matrixL();
      
      // det(h2 + hmin2) = det(L)^2,
      // donc sqrt(det(h2 + hmin2) = det(L)
      // donc log(sqrt(det(h2 + hmin2))) = 2log(det(L))/2 = log(det(L))
      // et log(det(h2 + hmin2)) = 2log(det(L))
      // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
      // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
      sum_log_diag_L = L.diagonal().array().log().sum();
      
      pen /= std::exp(sum_log_diag_L);
      
      // on résoud le systeme matriciel lineaire Lx = u
      x = L.colPivHouseholderQr().solve(extracted_grid.transpose());
      // den est un vecteur de taille n(n-1)/2
      den = x.array().square().colwise().sum();
      den = den.array() + 2 * sum_log_diag_L + d_log2pi;
      den *= - 0.5;
      den = den.array().exp();
      
      K -= 2 * den;
      
      loss = 2 * K.dot(extracted_bin_w);
      
    }
    else{
      // Calcul de la densite gaussienne multivariée K_{sqrt(2)h}
      // Pour un calcul plus rapide on passe par la decomposition de cholesky de 2*h2
      // Pour cela on utilise la librairie RcppEigen qui wrap la librairie c++ eigen
      
      // on recupere la matrice de la decomposition
      L = (2 * h2).llt().matrixL();
      
      // det(2*h2) = det(L)^2, donc log(det(2*h2)) = 2log(det(L))
      // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
      // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
      sum_log_diag_L = L.diagonal().array().log().sum();
      
      // on résoud le systeme matriciel lineaire Lx = u
      x = L.colPivHouseholderQr().solve(this->grid.transpose());
      // den est un vecteur de taille n(n-1)/2
      den = x.array().square().colwise().sum();
      den = den.array() + 2 * sum_log_diag_L + d_log2pi;
      den *= - 0.5;
      den = den.array().exp();
      
      K = den;
      
      
      L = (h2 + hmin2).llt().matrixL();
      
      // det(h2 + hmin2) = det(L)^2,
      // donc sqrt(det(h2 + hmin2) = det(L)
      // donc log(sqrt(det(h2 + hmin2))) = 2log(det(L))/2 = log(det(L))
      // et log(det(h2 + hmin2)) = 2log(det(L))
      // Comme L est diagonale, son determinant est egal au produit de ses elements diagonaux
      // donc le log de son determinant est egal a la somme des log de ses elements diagonaux
      sum_log_diag_L = L.diagonal().array().log().sum();
      
      pen /= std::exp(sum_log_diag_L);
      
      // on résoud le systeme matriciel lineaire Lx = u
      x = L.colPivHouseholderQr().solve(this->grid.transpose());
      // den est un vecteur de taille n(n-1)/2
      den = x.array().square().colwise().sum();
      den = den.array() + 2 * sum_log_diag_L + d_log2pi;
      den *= - 0.5;
      den = den.array().exp();
      
      K -= 2 * den;
      
      loss = 2 * K.dot(this->bin_w_inpairs_prod);
      
    }
    loss /= n2;
    double crit = loss + pen;
    return(crit);
  }
  
  
};







// [[Rcpp::export]]
Eigen::VectorXd h_GK_binned_mD_diag(Eigen::MatrixXd x_i, int nh_max, double tol, int nb_bin_per_axis=32, Rcpp::Nullable<Eigen::VectorXd> nb_bin_vect_ = R_NilValue){
  GK_binned_crit_mD_diag crit = GK_binned_crit_mD_diag(x_i, nb_bin_vect_, nb_bin_per_axis );
  Eigen::VectorXd h_opt = secdor_mD_diag(crit, nh_max, tol);
  return(h_opt);
}



// [[Rcpp::export]]
Eigen::MatrixXd h_GK_binned_mD_full(Eigen::MatrixXd x_i, Eigen::MatrixXd S, int nh_max, double tol, int nb_bin_per_axis=32, Rcpp::Nullable<Eigen::VectorXd> nb_bin_vect_ = R_NilValue){
  GK_binned_crit_mD_full crit = GK_binned_crit_mD_full(x_i, S, nb_bin_per_axis, nb_bin_vect_);
  Eigen::MatrixXd h_opt = secdor_mD_full(crit, nh_max, tol);
  return(h_opt);
}













