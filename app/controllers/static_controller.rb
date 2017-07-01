class StaticController < ActionController::Base # inherit from the base!
  include ApplicationHelper
  if Rails.env.production? || Rails.env.staging?
    after_filter :set_cache
  end
  layout false

  def show
    render template: "static/#{params[:page]}"
  end

  private

  def set_cache
    expires_in 3.minutes, public: true, must_revalidate: true
  end
end
