Rails.application.routes.draw do

  devise_for :users
  root to: 'root#index'

  post '/joke', to: 'root#get_joke'

  get 'static/:page' => 'static#show'
end
