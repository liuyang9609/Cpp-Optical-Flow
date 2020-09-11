void setupParameters(std::unordered_map<std::string, parameter> &parameters){
  parameter alpha = {"alpha", 8, 100, 1};
  parameter omega = {"omega", 195, 200, 100};
  parameter sigma = {"sigma", 10, 100, 10};
  parameter gamma = {"gamma", 990, 1000, 1000};
  parameter maxiter = {"maxiter", 30, 200, 1};
  parameter maxlevel = {"maxlevel", 4, 100, 1};
  parameter wrapfactor = {"wrapfactor", 95, 100, 100};
  parameter nonlinear_step = {"nonlinear_step", 3, 50, 1};
  parameter kappa = {"kappa", 100, 100, 100};
  parameter beta = {"beta", 150, 1000, 100};
  parameter deltat = {"deltat", 25, 100, 100};
  parameter phi_iter = {"phi_iter", 1, 100, 1};
  parameter iter_flow_before_phi = {"iter_flow_before_phi", 1, 100, 1};
  parameter Tm = {"Tm", 50, 100, 10};
  parameter Tr = {"Tr", 5, 20, 10};
  parameter Ta = {"Ta", 10, 800, 100};
  parameter blocksize = {"blocksize", 20, 100, 1};

  parameters.insert(std::make_pair<std::string, parameter>(alpha.name, alpha));
  parameters.insert(std::make_pair<std::string, parameter>(omega.name, omega));
  parameters.insert(std::make_pair<std::string, parameter>(sigma.name, sigma));
  parameters.insert(std::make_pair<std::string, parameter>(gamma.name, gamma));
  parameters.insert(std::make_pair<std::string, parameter>(maxiter.name, maxiter));
  parameters.insert(std::make_pair<std::string, parameter>(kappa.name, kappa));
  parameters.insert(std::make_pair<std::string, parameter>(beta.name, beta));
  parameters.insert(std::make_pair<std::string, parameter>(deltat.name, deltat));
  parameters.insert(std::make_pair<std::string, parameter>(phi_iter.name, phi_iter));
  parameters.insert(std::make_pair<std::string, parameter>(iter_flow_before_phi.name, iter_flow_before_phi));
  parameters.insert(std::make_pair<std::string, parameter>(wrapfactor.name, wrapfactor));
  parameters.insert(std::make_pair<std::string, parameter>(nonlinear_step.name, nonlinear_step));
  parameters.insert(std::make_pair<std::string, parameter>(maxlevel.name, maxlevel));
  parameters.insert(std::make_pair<std::string, parameter>(Tm.name, Tm));
  parameters.insert(std::make_pair<std::string, parameter>(Ta.name, Ta));
  parameters.insert(std::make_pair<std::string, parameter>(Tr.name, Tr));
  parameters.insert(std::make_pair<std::string, parameter>(blocksize.name, blocksize));
}

