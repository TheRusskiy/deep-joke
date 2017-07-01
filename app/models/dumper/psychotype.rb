module Dumper
  class Psyschotype < ::Dumper::Base

    def details
      {
          id: id,
      }
    end
  end
end
