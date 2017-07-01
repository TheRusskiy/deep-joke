class User < ApplicationRecord
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable and :omniauthable
  devise :database_authenticatable, :rememberable, :trackable, :validatable
         
  has_many :quiz_results
  has_many :quiz_requests
  has_many :identities
  
  validates_uniqueness_of :email

  def self.create_with_omniauth(auth)
    case auth.provider
    when 'twitter'
      create(
        first_name: auth.info.name,
        email: auth.info.email,
        password: Devise.friendly_token[0,20],
        twitter: auth.info.nickname
      )
    when 'facebook'
      create(
        first_name: auth.info.name,
        email: auth.info.email,
        password: Devise.friendly_token[0,20]
      )
    end
  end
         
  def name
    [self.first_name, self.last_name].join(" ")
  end
  
  def email_required?
    false
  end

  def email_changed?
    false
  end

  def in_json
    {
      id: id,
      name: name,
      quiz_result: quiz_results.last&.in_json,
      quiz_request: quiz_requests.last&.in_json,
    }
  end
end
