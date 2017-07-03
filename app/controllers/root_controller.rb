class RootController < ApplicationController
  protect_from_forgery with: :exception

  layout 'application'

  def index
  end

  def get_joke
    joke = JokeClient.predict(params[:text])[:text]
    joke = joke.downcase
    joke = joke.gsub('- ', '<br/>- ')
    symbols = %w[. , : ; _ - ! ? ( )]
    symbols.each do |symb|
      joke = joke.gsub(" #{symb}", symb)
    end
    joke = joke.gsub(/\s+/, ' ').strip
    render json: { joke: joke }
  end

  def current_user
    current_user
  end
end
