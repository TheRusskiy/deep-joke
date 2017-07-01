Warden::Manager.after_set_user do |user, auth, opts|
  scope = opts[:scope]
  auth.cookies.signed["user.id"] = user.id
  auth.cookies.signed["user.expires_at"] = 60.minutes.from_now
end

Warden::Manager.before_logout do |user, auth, opts|
  scope = opts[:scope]
  auth.cookies.signed["user.id"] = nil
  auth.cookies.signed["user.expires_at"] = nil
end