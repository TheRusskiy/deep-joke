class ApplicationController < ActionController::Base
  include Pundit
  protect_from_forgery with: :exception


  def after_sign_in_path_for(resource)
    stored_location_for(resource) || root_path
  end

  rescue_from UserError do |exception|
    respond_to do |format|
      error_message = exception.respond_to?(:message) ? exception.message : "Some error occurred!"
      data = exception.respond_to?(:data) ? exception.data : {}
      format.json do
        render json: {
            exception: exception.class.name,
            message: error_message,
            data: data
        }, status: 400
      end
      format.html do
        path = (request.referer && :back) || after_sign_in_path_for(current_user)
        redirect_to path, alert: error_message
      end
      format.js do
        render inline: "window.alert(#{error_message.to_json})".html_safe
      end
    end
  end

end
