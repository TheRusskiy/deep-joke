class BootstrapFormBuilder < ActionView::Helpers::FormBuilder
  # This is custom builder for bootstrap styled forms.
  # It's intended to be used with horizontal forms.
  # It has a lot of options and it's possible to use it with forms other than
  # horizontal, but I advice against it. Perhaps later if there's strong need I'll
  # create another one for other kinds of forms.
  #
  # To use it pass BootstrapFormBuilder to options as :builder parameter
  # and "form-horizontal" as :class.
  #
  # Sometimes you will find need to create a form for some information which
  # doesn't have corresponding column in database.
  # You have 2 options.
  # One is to create virtual field on model class that won't be
  # persisted into database.
  # Another is to create special class which you place into model.
  # This way you can add any validations on field and check object
  # validity via object.valid?. If there were errors you can render form
  # once again and builder will highlight fields with errors.
  #
  # BootstrapFormBuilder is integrated with jQuery validations plugin.
  # It can add validations but passing hash to :validate options.
  # You can specify masks via :mask option.
  # Builder accumulates all validations, masks etc and inserts them
  # into <script> tag after generated <form>.
  #
  # ==== Examples
  #   # usual usage
  #   form_for @type,
  #            url: psychos_profile_focus_path,
  #            builder: BootstrapFormBuilder,
  #            class:"form-horizontal" do |f|
  #    ...
  #   end
  #
  #  # custom class:
  #  class SomeModel
  #    # usual model stuff
  #    ...
  #    # following class isn't persisted to DB
  #    # but can be used in form_for
  #    class InfoAssociatedWithSomeModel
  #      include ActiveModel::Validations
  #      include ActiveModel::Conversion
  #      extend ActiveModel::Naming
  #      attr_accessor :method_name, :required_field
  #      validates_presence_of :required_field
  #
  #      def initialize(attributes = {})
  #        attributes.each do |name, value|
  #          send("#{name}=", value)
  #        end
  #      end
  #
  #      def persisted?
  #        false
  #      end
  #    end
  #  end
  #   form_for SomeModel::InfoAssociatedWithSomeModel.new, # or @info
  #            url: some_path,
  #            builder: BootstrapFormBuilder,
  #            class:"form-horizontal" do |f|
  #    ...
  #   end


  # This is how produced HTML looks:
=begin
<div class="form-group [has-error] field-container">
    <label for="@method_name" class="col-md-4 [control-label]">
      @label_text
    </label>
    <div class="col-md-6">
      @field
      <span class="help-inline has-error">
        @object.errors[method_name]
      </span>
    </div>
</div>
=end

=begin
# todo write spec for this case
# when options[:no_label]=true
<div class="form-group [has-error] field-container">
    <div></div>
    @field
    <span class="help-inline has-error">
      @object.errors[method_name]
    </span>
</div>
=end

  FIELD_CONTAINER_CLASS = "field-container" # not used for styling
  def initialize object_name, object, template, options
    options[:html]||={}
    html = options[:html]
    @form_class = options[:class]
    html[:class]="#{html[:class]} #{@form_class}"
    html[:role]="form"
    html[:autocomplete]||='off'
    html[:id]=options[:id] if options[:id]
    @builder_container = options[:parent_builder] || self
    if parent?
      @validations = {}
      html[:id]||="#{object_name}_#{object.object_id}"
      @parent_id = html[:id]
    end
    super object_name, object, template, options
  end

  # This method is invoked from monkey patched FormHelper (lies in lib directory)
  def generate_validation_script
    # http://jqueryvalidation.org/category/plugin/
    return "" if validations.nil? or validations.keys.length == 0
    script = %Q(
    <script>
      #{"$('##{@parent_id}').validate(#{validations_to_json(validations)});" if validations[:submitHandler] || validations[:rules]}
      #{generate_masks(validations[:masks]) if validations[:masks]}
    </script>
)
    script.html_safe
  end

  # ==== Options
  # This tag and others can accept any options usual tag would.
  # Options specific for this builder are specified below
  # validate:
  #    accepts hash with keys representing types of validations
  #    (see jquery validation plugin for provided validation types or define your own)
  #  mask:
  #    accepts String that will be passed to jQuery #mask method
  #  label:
  #    accepts params for label
  #    label[:empty] if true text of a label is replaced with invisible &zwnj; character
  #    if label is explicitly set to false e.g. label: false
  #    then label won't be generated at all and there won't be any columns,
  #    just an input field inside <div class="form-group"> <input ... /> </div>
  #
  # ==== Examples
  # # Generate text field for property :name and create label with text "Your name"
  # f.bs_text_field :name, "Your name"
  #
  # # Add validations
  # f.bs_text_field :dob, "Date of birth", validate: {required: true, date: true}
  #
  # # Add mask
  # f.bs_text_field :phone, "Your phone", mask: "(999) 999-9999"
  #
  # # Empty label
  # f.bs_text_field :name, "Your name", label: {empty: true}
  #
  # # No label and no label column
  # f.bs_text_field :name, "Your name", label: false
  #
  # # Usual Rails params
  # f.bs_text_field :name, "Your name", placeholder: "Enter your name here", value: "John Doe"
  def bootstrap_text_field(method_name, text = nil, options = {})
    switches = process_options(options, method_name)
    field= self.text_field method_name, options
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_text_field, :bootstrap_text_field

  # ==== Options
  # Same as bootstrap_text_field
  def bootstrap_text_area(method_name, text = nil, options = {})
    switches = process_options(options, method_name)
    field= self.text_area method_name, options
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_text_area, :bootstrap_text_area

  # ==== Options
  # Same as bootstrap_text_field
  def bootstrap_password_field(method_name, text = nil, options = {})
    switches = process_options(options, method_name)
    field= self.password_field method_name, options
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_password_field, :bootstrap_password_field

  # ==== Options
  # Same as bootstrap_text_field
  def bootstrap_file_field(method_name, text = nil, options = {})
    switches = process_options(options, method_name)
    field= self.file_field method_name, options
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_file_field, :bootstrap_file_field

  # ==== Options
  # Same as bootstrap_text_field
  # ==== Examples
  # f.bootstrap_select :state, 'State', states_array, prompt: "Select your state"
  # ...
  # def states_array
  #   [["Alaska", "AK"], ["Alabama", "AL"] ... ]
  # end
  # Note:
  # options and html_options are separate hashes, if you want to specify class, id etc.
  # you will have to use something like:
  # f.bootstrap_select :state, 'State', states_array, {prompt: "Select your state"}, id: 'some_id'
  def bootstrap_select(method_name, text = nil, values = [], options = {}, html_options = {})
    switches = process_options(options, method_name)
    switches2 = process_options(html_options, method_name)
    field= self.select method_name, values, options, html_options
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_select, :bootstrap_select

  # === Options
  # Same as bootstrap_text_field + some additional:
  # multiple:
  #   It's common express array attribute via multiple checkboxes.
  #   To do that pass "multiple: true" to options (not my invention, part of Rails).
  #   When you pass :multiple option both checkbox and label will be on the same column.
  #
  # inline:
  #   by passing this option elements checkbox and label are on the same
  #   column and the same line (inline-block)
  # === Examples
  #  f.bs_check_box :admin, "Has admin privileges"
  #
  #  f.bs_check_box :skill_ids,
  #                  skill.name,
  #                  {
  #                    multiple: true,
  #                    inline: true
  #                  },
  #                  skill.id, # value when checked
  #                  false     # value when unchecked
  def bootstrap_check_box(method_name, text = nil, options = {}, value="1", unchecked_value=0)
    options[:no_class] = true
    options[:label] = options[:label] || true
    if options[:multiple]
      multiple_value = rand(10000000)
      options[:id]=make_id "#{method_name}_#{multiple_value}"
      options[:multiple_value]=multiple_value
    end
    if options[:multiple] or options[:inline]
      options[:label] = {class:"", column:false}
      options[:field]||={}
      options[:field][:wrapper]||=false
      options[:field][:class]="#{options[:field][:class]} checkbox"
      # it's questionable whether putting style on element is a good idea
      # but it eliminates dependency on rule in stylesheet:
      options[:field][:style]= "display: inline-block" if options[:inline]
    else
      options[:class] = "checkbox"
    end
    switches = process_options(options, method_name)
    field= self.check_box method_name, options, value, unchecked_value
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_check_box, :bootstrap_check_box

  # Button for submitting form
  # === Options
  # Same as bootstrap_text_field + some additional:
  # submit_handler:
  #   this options allows to specify function that will be called
  #   when user submits form and the form is valid
  #   (part of validation library, see submitHandler at http://jqueryvalidation.org/category/plugin)
  #   allows to do some ajax and other custom form stuff.
  #   NOTE: function should NOT end with ";" (semicolon) as it will be passed to JSON
  # === Examples
  # f.bs_submit "Save changes"
  # f.bs_submit "Delete", class: "btn btn-danger
  # f.bs_submit "Submit Payment", id:"submit", submit_handler: %q(
  #                          function(form) {
  #                          $("#submit").attr('value', 'Processing your card. Please wait...');
  #                          $("#submit").attr('disabled', 'disabled');
  #                      })
  def bootstrap_submit(text, options = {})
    # todo submit_handler spec
    if text.is_a?(Hash)
      options = text
      text = "Save"
    end
    options[:no_class]=true if options[:no_class].nil?
    options[:class]||="btn btn-primary"
    if options[:submit_handler]
      validations[:submitHandler]=options[:submit_handler]
    end
    options[:id]||=@object_name.to_s+"_submit"
    options[:empty_label]= true unless options[:label] and options[:label][:empty]==false
    switches = process_options(options, text)
    field= self.submit text, options
    bootstrap_field(field, text, text, options.merge(switches)) + generate_validation_script
  end
  alias_method :bs_submit, :bootstrap_submit

  # Usually image_tag doesn't belong to any object
  # and accessed directly via:
  #   image_tag(args)
  # rather than via:
  #   form_for do |f| f.image_tag(args) end
  # But it's not possible to overwrite TagHelper as cleanly as FormBuilder
  # so it accessed via builder here.
  # I wanted to add this stuff to @template class, but was afraid to introduce memory
  # leaks by too smart metaprogramming.
  # === Options
  # Same as bootstrap_text_field
  # ==== Examples
  # f.bootstrap_image_tag @psycho.photo_url, "Current Profile Picture"
  # f.bootstrap_image_tag @psycho.photo_url, "Current Profile Picture", width: 150
  def bootstrap_image_tag(image_url, text = nil, options = {})
    options[:no_class]=true if options[:no_class].nil?
    switches = process_options(options, image_url)
    field= @template.image_tag image_url, options
    bootstrap_field(field, image_url, text, options.merge(switches))
  end
  alias_method :bs_image_tag, :bootstrap_image_tag

  # Sometimes you want to create a label with column distance consistent
  # with other fields, but create fields manually without this builder.
  # Or just place there whatever you want, like multiple checkboxes for example.
  # In this case you can you this method.
  # This method accepts block.
  # ==== Examples
  # Following will create column with label "Areas of focus" and column with multiple checkboxes
  #  <%= f.bs_custom :areas_of_focus, "Areas of focus" do %>
  #    <% Skill.all.each do |skill| %>
  #          <%= f.bs_check_box :skill_ids,
  #                             skill.name,
  #                             { multiple: true },
  #                             skill.id,
  #                             false %>
  #    <% end %>
  #  <% end %>
  #
  # If you don't specify any arguments except block label will have empty text:
  #   f.bs_custom do
  #      ...
  #   end
  def bootstrap_custom(method_name = nil, text = nil, options = {}, &block)
    if text.is_a?(Hash)
      options = text
    end
    if method_name.is_a?(Hash)
      options = method_name
      method_name = :no_field
    end
    if method_name.nil?
      options[:label]={empty:true}
      method_name = :no_field
    end
    switches = process_options(options, method_name)
    field= @template.capture(&block)
    bootstrap_field(field, method_name, text, options.merge(switches))
  end
  alias_method :bs_custom, :bootstrap_custom

  def bs_collection_check_boxes(*args)
    @template.collection_check_boxes(@object_name, *args) do |b|
      label_div = b.label(class: 'ts-label col-md-4 text-right') { b.text }
      value_div = @template.content_tag(:div, b.check_box(hidden: 'hidden') + b.label(class: 'ts-helper') { '' }, class: 'col-md-6 toggle-switch')
      @template.content_tag(:div, label_div + value_div, class: 'form-group')
    end
  end

  # todo bootstrap_radio

  ############# PROTECTED #########################################################
  protected
  # todo make sure it's safe
  def bootstrap_field(field, method_name, text = nil, options = {})
    text||=method_name.to_s.titleize
    label_class = "#{options[:label_column] && 'col-md-4'} #{ options[:label][:class] || 'control-label' }"
    label_text = options[:empty_label] ? @template.raw('&zwnj;') : text
    label=options[:no_label] ? "".html_safe : self.label(method_name, label_text, class: label_class, value: options[:multiple_value])
    if options[:validate]
      name=field.match(/name="(.*?)"/).captures[0]
      validations[:rules]||={}
      validations[:rules][name.to_s]=options[:validate]
    end
    if options[:mask]
      id=field.match(/id="(.*?)"/).captures[0]
      validations[:masks]||={}
      validations[:masks][id]=options[:mask]
    end
    errors = (@object.errors[method_name].present? and not options[:multiple]) ?
        @template.content_tag(:span,
                              @object.errors[method_name].join(', ').capitalize,
                              for: options[:id],
                              class:"help-inline has-error"
        )
        : ""
    # todo option for "no-group-margin" class (when embedded)
    field_wrapper=options[:no_label]||options[:no_field_wrapper] ?
        field+"\n"+errors
    : @template.content_tag(:div,
                            field+"\n"+errors,
                            class: options[:field][:class]||"col-md-6")
    if options[:no_reverse]
      i=1
    end
    @template.content_tag(:div,
                          (options[:inline] || options[:reverse]? "#{field_wrapper} \n #{label}" : "#{label} \n #{field_wrapper}").html_safe,
                          class: "#{options[:field][:class] && (options[:no_label]||options[:no_field_wrapper]) ? options[:field][:class] : 'form-group'} "+
                              "#{'has-error' if @object.errors[method_name].present?} "+FIELD_CONTAINER_CLASS,
                          style: options[:field][:style]
    )
  end

  # Access validations hash of top most builder.
  # be careful, used in specs via :send
  def validations
    parent? ? @validations : @builder_container.validations
  end

############## PRIVATE #########################################################
  private
  def process_options(options, method_name)
    switches = {}
    options[:label]={} if options[:label].nil? or options[:label]===true
    options[:field]={} if options[:field].nil?
    options[:id]||=make_id(method_name)
    if options[:label] === false
      options[:no_label]= true
      options[:label]={}
    end
    options[:no_field_wrapper] = true if options[:field] and (options[:field][:wrapper] === false)
    options[:empty_label] = true if options[:label] and options[:label][:empty]
    options[:label_column] = true unless options[:label] and options[:label][:column] === false
    options[:class]="#{options[:class]} form-control" unless options[:no_class]
    # don't send this stuff into field creation
    [:label,
     :validate,
     :submitHandler,
     :empty_label,
     :no_class,
     :no_label,
     :field,
     :multiple_value,
     :no_field_wrapper,
     :reverse,
     :inline,
     :label_column].each do |name|
      switches[name]=options[name]
      options[name]=nil
    end
    switches
  end

  def make_id(method_name)
    @object_name.to_s+'_'+method_name.to_s.downcase
  end

  def parent?
    @builder_container==self
  end

  def validations_to_json(validations)
    submit_handler = validations[:submitHandler] ? "submitHandler: #{validations[:submitHandler]}," : ""
    rules = validations[:rules] ? "rules: #{validations[:rules].to_json}," : ""
    %Q({
      #{rules}
    #{submit_handler}
      errorClass: "has-error",
      validClass: "",
//debug:true,
//onsubmit:false,
      errorElement: "span",
      highlight: #{highlight_function},
      unhighlight: #{unhighlight_function}
    })
  end

  def highlight_function
    %Q(
      function(element, errorClass, validClass) {
        $(element).closest(".#{FIELD_CONTAINER_CLASS}").addClass(errorClass).removeClass(validClass);
      }
    )
  end

  def unhighlight_function
    %Q(
      function(element, errorClass, validClass) {
        $(element).closest(".#{FIELD_CONTAINER_CLASS}").addClass(validClass).removeClass(errorClass);
      }
    )
  end

  def generate_masks masks
    masks.map do |key, value|
      "$('##{key}').mask('#{value}')"
    end * "; \n"
  end
end
