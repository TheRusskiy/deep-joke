# Monkeypatching Hash class
class Hash
  def get_deep(*fields)
    fields.inject(self) {|acc,e| acc[e] if acc}
  end

  def recursive_symbolize_keys!
    # Monkey patch for #deep_symbolize_keys in rails 4
    # https://grosser.it/2009/04/14/recursive-symbolize_keys/

    symbolize_keys!
    # symbolize each hash in .values
    values.each{|h| h.recursive_symbolize_keys! if h.is_a?(Hash) }
    # symbolize each hash inside an array in .values
    values.select{|v| v.is_a?(Array) }.flatten.each{|h| h.recursive_symbolize_keys! if h.is_a?(Hash) }
    self
  end

  def keys_to_snakecase
    result = self.map do |k, v|
      new_value = if v.respond_to?(:each_pair)
        v.respond_to?(:keys_to_snakecase) ? v.keys_to_snakecase : v
      elsif v.respond_to?(:each)
        v.map do |v1|
          v1.respond_to?(:keys_to_snakecase) ? v1.keys_to_snakecase : v1
        end
      else
        v
      end
      new_key = if k.is_a?(String)
        k.snakecase
      elsif k.is_a?(Symbol)
        k.to_s.snakecase.to_sym
      else
        k
      end
      [new_key, new_value]
    end.to_h
    if self.is_a?(HashWithIndifferentAccess)
      result = HashWithIndifferentAccess.new(result)
    end
    result
  end

  def keys_to_camelcase
    result = self.map do |k, v|
      new_value = if v.respond_to?(:each_pair)
        v.respond_to?(:keys_to_camelcase) ? v.keys_to_camelcase : v
      elsif v.respond_to?(:each)
        v.map do |v1|
          v1.respond_to?(:keys_to_camelcase) ? v1.keys_to_camelcase : v1
        end
      else
        v
      end
      new_key = if k.is_a?(String)
        k.camelize(:lower)
      elsif k.is_a?(Symbol)
        k.to_s.camelize(:lower).to_sym
      else
        k
      end
      [new_key, new_value]
    end.to_h
    if self.is_a?(HashWithIndifferentAccess)
      result = HashWithIndifferentAccess.new(result)
    end
    result
  end

  def to_indifferent_hash
    HashWithIndifferentAccess.new(self)
  end

end
