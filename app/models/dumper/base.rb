module Dumper
  class Base
    attr_accessor :entity, :return_nils

    def initialize(entity = nil)
      self.return_nils = entity.nil?
      self.entity = entity
    end

    def self.method_missing(name, *args, &block)
      name = name.to_s
      if new.respond_to?(name.gsub('each_', ''))
        if name.start_with?('each_')
          field_name = name.gsub('each_', '')
          args[0].map do |ent|
            new_dumper = new(ent)
            if new_dumper.return_nils
              return nil
            end
            result = new_dumper.send(field_name, *(args[1..-1]), &block)
            result.extend ShortKeysToCamelCase
            result
          end
        else
          new_dumper = new(args[0])
          if new_dumper.return_nils
            return nil
          end
          result = new_dumper.send(name, *(args[1..-1]), &block)
          result.extend ShortKeysToCamelCase
          result
        end
      else
        super name, *args, &block
      end
    end

    def self.respond_to_missing?(method_name, _ = false)
      new.respond_to? method_name
    end

    def method_missing(name, *args, &block)
      if entity.respond_to? name
        entity.send(name, *args, &block)
      else
        super name, *args, &block
      end
    end

    def respond_to_missing?(method_name, _ = false)
      entity.respond_to? method_name
    end

    module ShortKeysToCamelCase
      def camel
        keys_to_camelcase
      end
    end
  end
end
