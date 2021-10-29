#----------------------------------------------------------------------------------

#The codes for training autoencoders for categorical features and neural networks pre-trained with categorical autoencoders
#Based on the paper by £. Delong, A. Kozak, The use of autoencoders for training neural networks with mixed categorical and numerical features
#The paper is available on https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3952470

#----------------------------------------------------------------------------------

##Auxiliary functions

#----------------------------------------------------------------------------------

library(keras)

poisson_loss_function<-custom_metric("poisson_loss_function",function(y_true,y_pred){
  2*k_mean(y_true*k_log(y_true/y_pred)-(y_true-y_pred))})

function_loglikelihood<-function(args){
  c(log_probabilities,y_observations) %<-% args
  
  function_loglikelihood<-(
    k_sum(layer_multiply(list(y_observations,log_probabilities)),axis=2,keepdims = TRUE))  
}

max_loss_function <- custom_metric("max_loss_function",function(y_true,y_pred){k_mean(y_true-y_pred)})

min_max_scaler<-function(my_matrix){
  
  if (is.vector(my_matrix)==TRUE){
    
    arg_min=min(my_matrix)
    arg_max=max(my_matrix)
    
  }else{
    
    arg_min=apply(my_matrix,2,min)
    arg_max=apply(my_matrix,2,max)
  }
  
  if (is.vector(my_matrix)==TRUE){
    
    z=2*(my_matrix-arg_min)/(arg_max-arg_min)-1   
    
  }else{
    
    z=(2*((my_matrix-matrix(rep(arg_min,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE))/
            (matrix(rep(arg_max,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE)
             -matrix(rep(arg_min,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE)))-1)
  }
  
  return(z)
} 

#----------------------------------------------------------------------------------

##Data set

#----------------------------------------------------------------------------------

library(CASdatasets)
data("freMTPL2freq")
data_frequency=freMTPL2freq
data_frequency=data_frequency[sample(1:nrow(data_frequency),100000,replace=FALSE),]

#Response and numerical features

response=as.vector(data_frequency[,2]+10^(-7))

log_exposure=as.vector(log(data_frequency[,3]))

data_n=data_frequency[, c(7,9,11)]
data_n$VehGas <- ifelse(data_n$VehGas=="Regular",0,1)
data_n[,3]<-log(data_n[,3])
data_n<-min_max_scaler(data_n)

data_n=data.matrix(data_n)

#Categorical features and their ordinal coding

data=data_frequency[,c(4,5,6,8,10,12)]

data$VehPower[which(data$VehPower>9)] <- 9
data$VehPower <- data$VehPower-4

data$VehAge <- ifelse((data$VehAge>=0)&(data$VehAge<1),0,data$VehAge)
data$VehAge <- ifelse((data$VehAge>=1)&(data$VehAge<10),1,data$VehAge)
data$VehAge <- ifelse(data$VehAge>=10,2,data$VehAge)

data$DrivAge <- ifelse((data$DrivAge>=18)&(data$DrivAge<21),0,data$DrivAge)
data$DrivAge <- ifelse((data$DrivAge>=21)&(data$DrivAge<26),1,data$DrivAge)
data$DrivAge <- ifelse((data$DrivAge>=26)&(data$DrivAge<31),2,data$DrivAge)
data$DrivAge <- ifelse((data$DrivAge>=31)&(data$DrivAge<41),3,data$DrivAge)
data$DrivAge <- ifelse((data$DrivAge>=41)&(data$DrivAge<51),4,data$DrivAge)
data$DrivAge <- ifelse((data$DrivAge>=51)&(data$DrivAge<71),5,data$DrivAge)
data$DrivAge <- ifelse(data$DrivAge>=71,6,data$DrivAge)

for (i in c(4,5,6)){
  data[,i] <- match(data[,i],unique(data[,i]))-1
}

col_names=colnames(data)

#One-hot encoding of the categorical features

factors_one_hot=data.frame(to_categorical(data[,1]))
col_names_id=rep(col_names[1],ncol(factors_one_hot))
data_nn=factors_one_hot

for (i in (2:6)){
  factors_one_hot=data.frame(to_categorical(data[,i]))
  col_names_id=c(col_names_id,rep(col_names[i],ncol(factors_one_hot)))
  data_nn=data.frame(data_nn,factors_one_hot)
}

data_nn=data.matrix(data_nn)

#Add noise to your categorical data (if required by the user)

data_nn_noise=data_nn

#-------------------------------------------------------------------------------------

##Autoencoder softmax_all

#-------------------------------------------------------------------------------------

#Hyperparameters

no_neurons=8
epoch=100
batch_size=1000
learning_rate=0.001

#Network for the autoencoder

Input=layer_input(shape = c(ncol(data_nn)))

Output=Input %>% 
  layer_dense(units=no_neurons, activation='linear', use_bias=FALSE,name="encoder") %>% 
  layer_dense(units=ncol(data_nn), activation='softmax', use_bias=TRUE)

model_ae=keras_model(inputs=Input,outputs=Output)

#Optimize the cross entropy

model_ae %>% compile(optimizer=optimizer_nadam(lr=learning_rate),
                     loss="categorical_crossentropy")

CBs=list(callback_early_stopping(monitor="val_loss", min_delta=0,
                                  patience=15, verbose=0, mode=c("min"),
                                  restore_best_weights=TRUE))

network=model_ae %>% fit(data_nn_noise,data_nn,
                                  epochs=epoch,
                                  batch_size=batch_size,
                                  verbose=0,
                                  validation_data=list(data_nn_noise, data_nn),
                                  callbacks=CBs)

#Predict the output from the AE

predictions=model_ae %>% predict(data_nn)

#Recover the representation from the AE

encoder=keras_model(inputs=model_ae$input, outputs=get_layer(model_ae, "encoder")$output)
representation_categorical=encoder %>% predict(data_nn)

#------------------------------------------------------------------------------------

##Autoencoder softmax_per_feat

#------------------------------------------------------------------------------------

#Hyperparameters

no_neurons=8
epoch=100
batch_size=1000
learning_rate=0.001

#Network for the autoencoder

Input=layer_input(shape=c(ncol(data_nn)))

Output=Input %>% 
  layer_dense(units=no_neurons, activation='linear', use_bias=FALSE,name="encoder") %>% 
  layer_dense(units=ncol(data_nn), activation='linear', use_bias=TRUE)

i=1

dim_categorical=length(which(col_names_id==col_names[i]))
start_categorical=min(which(col_names_id==col_names[i]))

matrix_categorical=matrix(0,ncol(data_nn),dim_categorical)
matrix_categorical[start_categorical:(start_categorical+dim_categorical-1),]<-diag(1,dim_categorical)

Output_next=Output %>%
  layer_dense(units=c(dim_categorical),activation='softmax',
              trainable = FALSE,
              weights=list(array(matrix_categorical,dim=c(ncol(data_nn),dim_categorical)),
                           array(0,dim=c(dim_categorical))))

Output_final=Output_next

for (i in c(2:6)){
  
  dim_categorical=length(which(col_names_id==col_names[i]))
  start_categorical=min(which(col_names_id==col_names[i]))
  
  matrix_categorical=matrix(0,ncol(data_nn),dim_categorical)
  matrix_categorical[start_categorical:(start_categorical+dim_categorical-1),]<-diag(1,dim_categorical)
   
  Output_next=Output %>%
    layer_dense(units=c(dim_categorical),activation='softmax',
                trainable = FALSE,
                weights=list(array(matrix_categorical,dim=c(ncol(data_nn),dim_categorical)),
                             array(0,dim=c(dim_categorical))))
  
  Output_final=list(Output_final,Output_next)%>%layer_concatenate              
}

Output_final <- keras_model(inputs=c(Input),outputs=Output_final)

#Network for the log probabilities

inputs=layer_input(shape=c(ncol(data_nn)))

log_transform_probabilities=inputs %>%
  layer_dense(units=ncol(data_nn),activation='linear',trainable=FALSE,
              weights=list(array(diag(ncol(data_nn)),dim=c(ncol(data_nn),ncol(data_nn))),
                           array(10^(-7),dim=c(ncol(data_nn))))) %>%
  layer_dense(units=ncol(data_nn),activation=k_log,trainable=FALSE,
              weights=list(array(diag(ncol(data_nn)),dim=c(ncol(data_nn),ncol(data_nn))),
                           array(0,dim=c(ncol(data_nn)))))

log_transform_probabilities=keras_model(inputs=inputs,outputs=log_transform_probabilities)

#Network for the log-likelihood

Input=layer_input(shape=c(2*ncol(data_nn)))
Input_noise=layer_lambda(Input,f=function(x){Input[,1:(ncol(data_nn)),drop=FALSE]})  
Input_true=layer_lambda(Input,f=function(x){Input[,(ncol(data_nn)+1):(2*ncol(data_nn)),drop=FALSE]}) 

log_probabilities=log_transform_probabilities(Output_final(Input_noise))
function_loglikelihood=layer_lambda(list(log_probabilities,Input_true),
                       function_loglikelihood)

model_optimize=keras_model(inputs=Input, outputs=function_loglikelihood)

#Optimize the loglikelihood

model_optimize %>% compile(optimizer=optimizer_nadam(lr=learning_rate),loss=max_loss_function)

CBs=list(callback_early_stopping(monitor = "val_loss", min_delta = 0,
                                   patience = 15, verbose = 0, mode = c("min"),
                                   restore_best_weights = TRUE))

X_all=data.matrix(cbind(data_nn_noise, data_nn))
Y_all=as.vector(rep(0,nrow(data_nn)))

network=model_optimize %>% fit(X_all,Y_all,
                                  epochs=epoch,
                                  batch_size=batch_size,
                                  verbose=0,
                                  validation_data=list(X_all,Y_all),
                                  callbacks=CBs)

#Predict the output from the AE

predictions=Output_final %>% predict(data_nn)

#Recover the representation from the AE

encoder=keras_model(inputs=Output_final$input, outputs=get_layer(Output_final, "encoder")$output)
representation_categorical=encoder %>% predict(data_nn)

#------------------------------------------------------------------------------------

##Update the weights in the encoder

#------------------------------------------------------------------------------------

weights_ae=matrix(unlist(get_weights(encoder)[1]),ncol=no_neurons, byrow = FALSE) 

representation_categorical=encoder %>% predict(data_nn)
arg_min=apply(representation_categorical,2,min)
arg_max=apply(representation_categorical,2,max)

a=2/(arg_max-arg_min)
b=(-1-2*arg_min/(arg_max-arg_min))/sum(data_nn[1,])
weights_ae=t(t(weights_ae)*a+b)

z=get_weights(encoder)
z[[1]]=weights_ae
set_weights(encoder,z)

#-----------------------------------------------------------------------------------

##Neural network pre-trained with categorical autoencoder

#-----------------------------------------------------------------------------------

#Hyperparameters

neurons_layer_1=20
neurons_layer_2=15
neurons_layer_3=10

learning_rate_nn=10^(-4)
epoch_nn=1000
batch_size_nn=1000

#Training, validation and test sets

sample_train=c(1:60000)
sample_val=c(60001:80000)
sample_test=c(80001:100000)

X_num_train=data_n[sample_train,]
X_num_val=data_n[sample_val,]
X_num_test=data_n[sample_test,]

X_cat_train=data_nn[sample_train,]
X_cat_val=data_nn[sample_val,]
X_cat_test=data_nn[sample_test,]

response_train=response[sample_train]
response_val=response[sample_val]
response_test=response[sample_test]

log_exposure_train=log_exposure[sample_train]
log_exposure_val=log_exposure[sample_val]
log_exposure_test=log_exposure[sample_test]

#Network
#One can add the 2nd autoencoder for the first hidden layer (see the paper)

Input_cat <- layer_input(shape=c(ncol(X_cat_train)))
Input_num <- layer_input(shape=c(ncol(X_num_train)))
LogExp <- layer_input(shape=c(1))

Output_cat=Input_cat %>% 
    layer_dense(units=no_neurons, activation='linear',
                trainable = TRUE,use_bias=FALSE,
                weights=list(array(weights_ae,dim=c(ncol(X_cat_train),no_neurons))))

Input <- list(Output_cat,Input_num) %>% layer_concatenate

Output=Input %>% 
    layer_dense(units=neurons_layer_1, activation='tanh') %>% 
    layer_dense(units=neurons_layer_2, activation='tanh') %>%
    layer_dense(units=neurons_layer_3, activation='tanh') %>% 
    layer_dense(units=1, activation='linear')
  
Output=list(Output,LogExp) %>% layer_add %>%
   layer_dense(units=1, activation=k_exp, trainable=FALSE ,
               weights=list(array(1, dim=c(1,1)), array(0,dim=c(1))))

model_nn <- keras_model(inputs=list(Input_cat,Input_num,LogExp),outputs=Output)

#Optimize the Poisson loss function

model_nn %>% compile(optimizer=optimizer_nadam(lr=learning_rate_nn),loss=poisson_loss_function)

CBs <-list(callback_early_stopping(monitor="val_loss", min_delta=0,
                                   patience=15, verbose=0, mode=c("min"),
                                   restore_best_weights=TRUE))

network <- model_nn %>% fit(list(X_cat_train,X_num_train,log_exposure_train),
                             response_train, 
                             epochs=epoch_nn,
                             batch_size=batch_size_nn,
                             verbose=0,
                             validation_data=list(list(X_cat_val,X_num_val,log_exposure_val),response_val),
                             callbacks=CBs)

#Deviance on the test set

predictions <- model_nn %>% predict(list(X_cat_test,X_num_test,log_exposure_test))

2*mean(response_test*log(response_test/predictions)-(response_test-predictions))
