# Exception handler
class PaymentError < StandardError
  attr_accessor :data
  def initialize message, data = nil
    @data = data
    super(message)
  end
end
