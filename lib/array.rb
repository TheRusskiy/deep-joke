# Monkeypatch for Array class
class Array
  def in_sql(quote: "'")
    map { |a| "#{quote}#{a}#{quote}" }.join(',')
  end

  def keys_to_snakecase
    map { |e| e.respond_to?(:keys_to_snakecase) ? e.keys_to_snakecase : e }
  end

  def keys_to_camelcase
    map { |e| e.respond_to?(:keys_to_camelcase) ? e.keys_to_camelcase : e }
  end
end
