require_relative 'boot'

require 'rails/all'

# Require the gems listed in Gemfile, including any gems
# you've limited to :test, :development, or :production.
Bundler.require(*Rails.groups)

module Psycho
  class Application < Rails::Application
    # Settings in config/environments/* take precedence over those specified here.
    # Application configuration should go into files in config/initializers
    # -- all .rb files in that directory are automatically loaded.
    config.time_zone = 'Pacific Time (US & Canada)'
    config.assets.paths << Rails.root.join('node_modules')
    config.assets.paths << Rails.root.join('bower_components')
    require 'email_validator'
    require 'hash'
    require 'array'
    require 'bootstrap_form_builder'
    require 'user_error'
    require 'payment_error'


    config.action_cable.mount_path = '/cable'

    config.lograge.enabled = true
    config.lograge.custom_options = lambda do |event|
      stacktrace = {}
      strace = event.payload[:stacktrace]
      if strace
        quoted_stacktrace = %('#{Array(strace).reject{|l| l.include?('/gems/') || l.include?('/instrumenter')}.join("\n\t")}') # depending on if I'm logging as logstash or not
        stacktrace = {stacktrace: quoted_stacktrace}
      end
      exceptions = %w(controller action format id)
      {
          id: event.payload[:ip],
          user_id: event.payload[:user_id],
          params: event.payload[:params].except(*exceptions)
      }.merge(stacktrace)
    end

  end
end
