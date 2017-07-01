Rails.application.config.middleware.use OmniAuth::Builder do
  on_failure { |env| AuthenticationsController.action(:failure).call(env) }
  configure do |config|
    config.path_prefix = '/auth'
  end
end