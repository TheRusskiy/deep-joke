class EmployeeOrdersChannel < ApplicationCable::Channel
  # Called when the consumer has successfully
  # become a subscriber of this channel.
  def subscribed
    stream_from "employee_restaurant_#{current_user.restaurant_id}_orders"
  end
  
  def speak(data)
    ActionCable.server.broadcast 'test', message: data['message']
  end
end