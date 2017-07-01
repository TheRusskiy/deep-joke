class JokeClient
  def self.predict(text)
    use_ssl = false
    uri = URI.parse("http#{use_ssl ? 's' : ''}://localhost:6000/get-joke")
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = use_ssl
    request = Net::HTTP::Post.new(uri.request_uri, {'Content-Type' =>'application/json'})
    body = {
      text: text,
    }
    request.body = body.to_json
    response = http.request(request)
    result = HashWithIndifferentAccess.new JSON.parse(response.body)
    ap result
    result
  end
end