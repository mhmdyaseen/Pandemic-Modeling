using MAT
using Plots, Measures, Lux
using DifferentialEquations
using DiffEqFlux, Optimization, OptimizationFlux
using Random

#import data and cleanup
vars = matread("Rise_Italy_Track.mat")
Random.seed!(50)

Infected = vars["Italy_Infected_All"]
Recovered = vars["Italy_Recovered_All"]
Dead= vars["Italy_Dead_All"]
Time = vars["Italy_Time"]

Infected = Infected - Recovered - Dead

#define UDE
rng = Random.default_rng()
ann = Lux.Chain(Lux.Dense(3,10,relu),Lux.Dense(10,1))
p1, st1 = Lux.setup(rng, ann)

parameter_array = Float64[0.15, 0.013, 0.01]

p0_vec = (layer_1 = p1, layer_2 = parameter_array)
p0_vec = ComponentArray(p0_vec)

#QSIR model ODE with UDE neural network
function QSIR(du, u, p, t)
    β = abs(p.layer_2[1])
    γ = abs(p.layer_2[2])
    δ = abs(p.layer_2[3])
    #γ = abs(γ_parameter)
    #δ = abs(δ_parameter)

    UDE_term = abs(ann([u[1]; u[2]; u[3]], p.layer_1, st1)[1][1])

    du[1]=  - β*u[1]*(u[2])/u0[1]
    du[2] = β*u[1]*(u[2])/u0[1] - γ*u[2] - UDE_term*u[2]/u0[1]
    du[3] = γ*u[2] + δ*u[4]
    du[4] =  UDE_term*u[2]/u0[1] - δ*u[4]
end

α = p0_vec

#problem setup
u0 = Float64[60000000.0, 593, 62, 10]
tspan = (0, 95.0)
datasize = 95;

prob= ODEProblem{true}(QSIR, u0, tspan)
t = range(tspan[1],tspan[2],length=datasize)

#defining the loss function and performing the UDE training
function predict_adjoint(θ) # Our 1-layer neural network
    x = Array(solve(prob,Tsit5(),p=θ,saveat=t,
                  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss_adjoint(θ)
      prediction = predict_adjoint(θ)
      loss = sum(abs2, log.(abs.(Infected[1:end])) .- log.(abs.(prediction[2, :] .+ prediction[4, :] ))) + (sum(abs2, log.(abs.(Recovered[1:end] + Dead[1:end]) ) .- log.(abs.(prediction[3, :] ))))
      return loss
    end

#callback to monitor optimization progress
iter = 0
function callback3(θ, l)
    global iter
    iter += 1
    if iter % 10 == 0
        println("Iteration $iter: Loss = $l")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
optprob = OptimizationProblem(optf, p0_vec)

#ADAM optimizer
res1 = Optimization.solve(optprob, ADAM(0.01), callback=callback3, maxiters=15000)

#BFGS fine-tuning
optprob2 = remake(optprob, u0=res1.u)
res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01),
                          callback=callback3, maxiters=100)

p_trained = res2.u
data_pred = predict_adjoint(p_trained)

#extract SIR-T predicted values

S_NN_all_loss = data_pred[1, :]
I_NN_all_loss = data_pred[2, :]
R_NN_all_loss = data_pred[3, :]
T_NN_all_loss = data_pred[4, :]

#compute learned quarantine function Q(t)
Q_parameter = zeros(Float64, length(S_NN_all_loss), 1)

for i = 1:length(S_NN_all_loss)
  Q_parameter[i] = abs(ann([S_NN_all_loss[i];I_NN_all_loss[i]; R_NN_all_loss[i]], p3n.layer_1, st1)[1][1])
end

#plotting
#plot 1: data prediction
bar(Infected',alpha=0.5,label="Data: Infected",color="red")
plot!(t, data_pred[2, :] .+ data_pred[4, :] , xaxis = "Days post 500 infected", label = "Prediction", legend = :topright, framestyle = :box, left_margin = 5mm, bottom_margin = 5mm, top_margin = 5mm,  grid = :off, color = :red, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing, yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)
bar!(Recovered' + Dead',alpha=0.5,xrotation=60,label="Data: Recovered", color="blue")
plot!(t, data_pred[3, :], ylims = (-0.05*maximum(Recovered + Dead),1.5*maximum(Recovered + Dead)), right_margin = 5mm, xaxis = "Days post 500 infected", label = "Prediction ", legend = :topleft, framestyle = :box, left_margin = 5mm, bottom_margin =5mm, top_margin = 5mm, grid = :off, color = :blue, linewidth  = 4, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman", legendfontsize = 1))

#plot 2: Q(t) compared with when lockdown was imposed in Italy.
scatter(t,Q_parameter/u0[1], xlims = (0, datasize+1), ylims = (0,1), xlabel = "Days post 500 infected", ylabel = "Q(t)", label = "Quarantine strength",color = :black, framestyle = :box, grid =:off, legend = :topleft, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12)

D1 = diff(Q_parameter, dims = 1)
D2 = diff(D1, dims = 1)
Transitionn = findall(x -> x <0, D2)[1]

#highlight lockdown & inflection point
plot!([11-0.01,11+0.01],[0.0, 0.6],lw=3,color=:green,label="Government Lockdown imposed",linestyle = :dash)
plot!([Int(Transitionn[1])-0.01,Int(Transitionn[1])+0.01],[0.0, 0.6],lw=3,color=:red,label="Inflection point in learnt Q(t)",linestyle = :dash)
