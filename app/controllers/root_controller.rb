class RootController < ApplicationController
  protect_from_forgery with: :exception

  layout 'application'

  def index
  end

  def get_joke
    joke = JokeClient.predict(params[:text])
    # joke = "haha"
    render json: { joke: joke[:text].downcase }
  end

  def current_user
    current_user
  end
end
