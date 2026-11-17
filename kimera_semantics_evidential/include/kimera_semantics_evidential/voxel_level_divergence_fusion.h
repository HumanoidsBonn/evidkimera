#include <Eigen/Dense>
#include <cmath>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <kimera_semantics_evidential/voxel_level_fusion_base.h>
#include <kimera_semantics_evidential/semantic_voxel.h>

class DivergenceWeightedFusion : public VoxelLevelFusionBase {
 public:
  explicit DivergenceWeightedFusion(bool replace_on_first_measurement = true)
      : VoxelLevelFusionBase(replace_on_first_measurement) {}

  void fuse(SemanticVoxel* voxel,
            const voxblox::SemanticState& measurement,
            float tsdf_weight = 1.0f,
            float measurement_uncertainty = 1.0f) const override {
    // Handle the first measurement case
    handleFirstMeasurement(voxel, measurement);

    // Calculate total evidence (sum of Dirichlet parameters)
    float S_prior = voxel->dirichlet_params.sum();
    float S_new = measurement.sum();

    // If the new measurement has higher evidence, compute divergence
    if (S_new > S_prior/voxel->num_measurements) {
      // Compute KL divergence between prior and new measurement
      double divergence = computeJensenShannonDivergence(voxel->dirichlet_params, measurement);

      // Compute weighting factor based on divergence and new evidence
      float weight = computeWeight(divergence, S_new);

      // Update the voxel's Dirichlet parameters with weighted fusion
      voxel->dirichlet_params = (1.0f - weight) * voxel->dirichlet_params + weight * tsdf_weight*measurement;
    } else {
      // If new evidence is not higher, perform standard fusion
      voxel->dirichlet_params += tsdf_weight*measurement;
    }

    // Transition the voxel's update state
    transitionUpdateState(voxel);
  }

  bool requiresMeasurementUncertainty() const override { return false; }

 protected:
  double computeKLDivergence(const Eigen::VectorXf& alpha_prior,
                             const Eigen::VectorXf& alpha_new) const {
    float sum_alpha_prior = alpha_prior.sum();
    float sum_alpha_new = alpha_new.sum();

    double log_B_prior = std::lgamma(sum_alpha_prior) - alpha_prior.unaryExpr(&std::lgamma).sum();
    double log_B_new = std::lgamma(sum_alpha_new) - alpha_new.unaryExpr(&std::lgamma).sum();

    Eigen::VectorXf digamma_prior = alpha_prior.unaryExpr(&boost::math::digamma);
    float digamma_sum_prior = boost::math::digamma(sum_alpha_prior);

    double kl_div = log_B_new - log_B_prior +
                    (alpha_prior - alpha_new).dot(digamma_prior - Eigen::VectorXf::Constant(alpha_prior.size(), digamma_sum_prior));

    return kl_div;
  }

  double computeBhattacharyyaDistance(const Eigen::VectorXf& alpha_prior,
                                      const Eigen::VectorXf& alpha_new) const {
    Eigen::VectorXf alpha_mid = 0.5 * (alpha_prior + alpha_new);
    double log_B_prior = std::lgamma(alpha_prior.sum()) - alpha_prior.unaryExpr(&std::lgamma).sum();
    double log_B_new = std::lgamma(alpha_new.sum()) - alpha_new.unaryExpr(&std::lgamma).sum();
    double log_B_mid = std::lgamma(alpha_mid.sum()) - alpha_mid.unaryExpr(&std::lgamma).sum();

    return log_B_mid - 0.5 * (log_B_prior + log_B_new);
  }

  double computeHellingerDistance(const Eigen::VectorXf& alpha_prior,
                                  const Eigen::VectorXf& alpha_new) const {
    Eigen::VectorXf alpha_mid = 0.5 * (alpha_prior + alpha_new);
    double log_B_prior = std::lgamma(alpha_prior.sum()) - alpha_prior.unaryExpr(&std::lgamma).sum();
    double log_B_new = std::lgamma(alpha_new.sum()) - alpha_new.unaryExpr(&std::lgamma).sum();
    double log_B_mid = std::lgamma(alpha_mid.sum()) - alpha_mid.unaryExpr(&std::lgamma).sum();

    return std::sqrt(1.0 - std::exp(log_B_mid - 0.5 * (log_B_prior + log_B_new)));
  }
  double computeJensenShannonDivergence(const Eigen::VectorXf& alpha_prior, const Eigen::VectorXf& alpha_new) const {
    
    Eigen::VectorXf alpha_mid = 0.5 * (alpha_prior + alpha_new);
    // Compute log Beta function values
    double log_B_prior = std::lgamma(alpha_prior.sum()) - alpha_prior.unaryExpr(&std::lgamma).sum();
    double log_B_new = std::lgamma(alpha_new.sum()) - alpha_new.unaryExpr(&std::lgamma).sum();
    double log_B_mid = std::lgamma(alpha_mid.sum()) - alpha_mid.unaryExpr(&std::lgamma).sum();
    // Compute KL divergences
    double KL_prior_mid = log_B_mid - log_B_prior +
    ((alpha_prior.array() - alpha_mid.array()) *
    (alpha_prior.unaryExpr(&std::digamma).array() -
    std::digamma(alpha_prior.sum()))).sum();
    double KL_new_mid = log_B_mid - log_B_new +
    ((alpha_new.array() - alpha_mid.array()) *
    (alpha_new.unaryExpr(&std::digamma).array() -
    std::digamma(alpha_new.sum()))).sum();
    // JS divergence
    return 0.5 * (KL_prior_mid + KL_new_mid);
}

  float computeWeight(double divergence, float S_new, float lambda = 0.5f) const {
    return 1.0f - divergence;
  }
};
